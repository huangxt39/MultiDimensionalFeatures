"""
For a given layer and cluster, creates an interactive plotly plot
of the cosine simlarities between the SAE decoder features in the 
cluster and also PCA projections of the reconstructed
activations with just the features in the cluster being allowed
to fire.
"""

import os
import time
import pickle
import argparse
import re
import torch
from pathlib import Path
import json

# hopefully this will help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["TRANSFORMERS_CACHE"] = "/om/user/ericjm/.cache/"

import einops
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sae_lens import SAE
# import transformer_lens
from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.decomposition import PCA
import plotly.subplots as sp
import plotly.graph_objects as go

from utils import BASE_DIR

torch.set_grad_enabled(False)

def get_gpt2_sae(device, layer):
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device=device
    )[0]

def get_cluster_activations(sparse_sae_activations, sae_neurons_in_cluster, decoder_vecs, sample_limit):
    current_token = None
    all_activations = []
    all_token_indices = []
    updated = False
    pbar = tqdm(total=sample_limit)
    for sae_value, sae_index, token_index in zip(
        sparse_sae_activations["sparse_sae_values"],
        sparse_sae_activations["sparse_sae_indices"],
        sparse_sae_activations["all_token_indices"],
    ):
        if current_token == None:
            current_token = token_index
            current_activations = torch.zeros(768)
        if token_index != current_token:
            if updated:
                all_activations.append(current_activations)
                all_token_indices.append(token_index)
                pbar.update(1)
                if sample_limit is not None and len(all_activations) == sample_limit:
                    break
            updated = False
            current_token = token_index
            current_activations = torch.zeros(768)
        if sae_index in sae_neurons_in_cluster:
            updated = True
            current_activations += sae_value * decoder_vecs[sae_index]
    pbar.close()
    if all_activations:
        return torch.stack(all_activations), all_token_indices
    else:
        return all_activations, all_token_indices

def get_R(args, clusters_file, layer):
    
    ae = get_gpt2_sae(device="cpu", layer=layer)
    decoder_vecs = ae.W_dec.data.cpu()

    sparse_sae_activations = np.load(f"{BASE_DIR}gpt-2/sae_activations_big_layer-{layer}.npz")

    # load up clusters
    with open(clusters_file, "rb") as f:
        clusters = pickle.load(f)
    
    clusters_with_order = []
    for cluster in clusters:
        cluster_features = decoder_vecs[cluster]
        cos_sims = cluster_features @ cluster_features.T
        cos_sims.fill_diagonal_(0)
        clusters_with_order.append((cluster, len(cluster), cos_sims.mean().item()))

    if args.order_by == "avg_sim":
        print("order cluster by average similarity")
        clusters_with_order.sort(key=lambda x: -x[-1])
    else:
        print("order cluster by size")
        clusters_with_order.sort(key=lambda x: -x[1])
    
    # for rtol in [0.005, 0.01, 0.02]:
    rtol = args.rtol
    R = torch.zeros((0, decoder_vecs.size(1)))
    R_config = []
    for i, (cluster, cluster_size, avg_cos_sim) in enumerate(clusters_with_order):
        print("new cluster", i, cluster_size, avg_cos_sim)
        if cluster_size < 4:
            print("too small, skip")
            continue
        reconstructions, _ = get_cluster_activations(sparse_sae_activations, set(cluster), decoder_vecs, args.sample_limit)
        if len(reconstructions) == 0:
            print("no reconstructions, skip")
            continue
        # rm variance in previous saved basis
        if R.size(0) > 0:
            reconstructions -= reconstructions @ R.T @ R

        # pca = PCA()
        # pca.fit(reconstructions.numpy())
        # sklearn_results = pca.components_

        reconstructions -= reconstructions.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(reconstructions)
        print("max singular val", S[0].item())
        if S[0].item() < 1e-6:
            print("variance already taken by prev subspaces, skip")
            continue

        thr = S[0] * rtol
        num_comp = (S > thr).sum().item()
        new_basis = Vh[: num_comp]

        R = torch.cat([R, new_basis], dim=0)
        R_config.append(num_comp)
        print("new R size", R.size())

        if R.size(0) == R.size(1) or len(R_config) == args.max_n_subspace:
            break

        # print("imple diff", (Vh[: num_comp] - torch.from_numpy(sklearn_results)[: num_comp]).abs().mean())

    if R.size(0) < R.size(1):
        U, _, _ = torch.linalg.svd(R.T)
        new_basis = U[:, R.size(0):].T
        R = torch.cat([R, new_basis], dim=0)
        R_config.append(new_basis.size(0))
    # !! important, R.T is the R used in paper's repo

    # temp = torch.eye(R.size(0))
    # print("error to identity, mean", (R @ R.T - temp).abs().mean(), "max", (R @ R.T - temp).abs().max())
    print("final config", R_config)
    return R.T, R_config


if __name__ == '__main__':
    # rtol = 0.01, 0.03, 0.1
    # order_by = avg_sim, "size"
    parser = argparse.ArgumentParser(description="Create cluster figure")
    parser.add_argument("--sample_limit", type=int, help="Max number of reconstructions in plot", default=20_000)
    parser.add_argument("--rtol", type=float, help="relative threshold for computing the rank", default=0.01)
    parser.add_argument("--order_by", type=str, help="order cluster by", default="avg_sim", choices=["avg_sim", "size"])
    parser.add_argument("--max_n_subspace", type=int, help="maximum num of subspaces we want", default=50)
    args = parser.parse_args()

    convert_config = f"order_by-{args.order_by}-rtol-{args.rtol}"
    root_save_dir = Path(BASE_DIR).parent / "trainedRs"
    if not root_save_dir.exists():
        root_save_dir.mkdir()

    cluster_dir = Path(BASE_DIR).parent / "clusters"
    paths = os.listdir(cluster_dir)
    cluster_configs = set([re.search(r"_clusters_(.+)\.pkl", p).group(1) for p in paths])
    layers = [5, 7, 9, 10]
    for cluster_config in cluster_configs:
        keep_config = True
        for layer in layers:
            clusters_file = f"gpt-2_layer_{layer}_clusters_{cluster_config}.pkl"
            if not (cluster_dir / clusters_file).exists():
                keep_config = False
                break
            with open(cluster_dir / clusters_file, "rb") as f:
                if len(pickle.load(f)) < 5:
                    keep_config = False
                    break

        if keep_config:
            print(cluster_config)
            save_dir = root_save_dir / f"gpt2-sae_feature-cluster_cfg-{cluster_config}-{convert_config}"
            if save_dir.exists():
                print("skip: already exists")
                continue
            layer_to_R = {}
            for layer in layers:
                clusters_file = f"gpt-2_layer_{layer}_clusters_{cluster_config}.pkl"
                R, R_config = get_R(args, cluster_dir / clusters_file, layer)
                layer_to_R[layer] = (R, R_config)
                if len(R_config) > args.max_n_subspace:
                    break

            if all(5 <= len(layer_to_R[layer][1]) <= args.max_n_subspace for layer in layer_to_R):
                if not save_dir.exists():
                    save_dir.mkdir()
                for layer in layers:
                    R, R_config = layer_to_R[layer]
                
                    with open(save_dir / f"R_config-gpt2-x{layer-1}.post.json", "w") as f:
                        # x(n-1).post = x(n).pre
                        json.dump({"partition": R_config}, f)
                    
                    torch.save({"R.parametrizations.weight.0.base": R}, save_dir / f"R-gpt2-x{layer-1}.post.pt")
                print("subspaces saved at", str(save_dir))
            else:
                print("skip: num subspace out of bound", cluster_config, convert_config, "\t num subspaces:", [len(layer_to_R[layer][1]) for layer in layers])