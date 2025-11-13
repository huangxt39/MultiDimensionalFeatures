"""
Saves a list of list of indices of SAE features as a pickle file.
"""

from utils import get_sae, BASE_DIR
import torch
from tqdm import tqdm
import pickle
import argparse
from sklearn.cluster import SpectralClustering
import numpy as np
import os
from pathlib import Path
import re

torch.set_grad_enabled(False)

device = "cpu"


def spectral_cluster_sims(all_sims, n_clusters):
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
    labels = sc.fit_predict(all_sims).tolist()
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(i)
    return clusters
    

def graph_cluster_sims(
    all_sims, top_k_for_graph=2, sim_cutoff=0.5, prune_clusters=False
):
    near_neighbors = torch.topk(all_sims, top_k_for_graph, dim=1)

    graph = [[] for _ in range(all_sims.shape[0])]
    for i in tqdm(range(all_sims.shape[0])):
        top_indices = near_neighbors.indices[i]
        top_sims = near_neighbors.values[i]
        top_indices = top_indices[top_sims > sim_cutoff]
        graph[i] = top_indices.tolist()

    # Add back edges
    for i in tqdm(range(all_sims.shape[0])):
        for j in graph[i]:
            if i not in graph[j]:
                graph[j].append(i)

    # Find connected components
    visited = [False] * all_sims.shape[0]
    components = []
    for i in range(all_sims.shape[0]):
        if visited[i]:
            continue
        component = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            stack.extend(graph[node])
        components.append(component)

    if prune_clusters:
        threshold = 1000
        components = [c for c in components if len(c) < threshold and len(c) > 1]

    return components



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-2",
                    choices=["mistral", "gpt-2"])
parser.add_argument(
    "--method", type=str, choices=["graph", "spectral"], default="graph"
)
parser.add_argument("--include_only_first_k_sae_features", type=int)
args = parser.parse_args()

layers = [5, 7, 9, 10]

root_save_dir = Path(BASE_DIR).parent / "clusters"
if not root_save_dir.exists():
    root_save_dir.mkdir()


for layer in layers:
    model_name = args.model_name
    method = args.method

    if model_name == "mistral":
        model_name = "mistral-7b"


    sae = get_sae(device=device, model_name=model_name, layer=layer)
    all_sae_features = sae.W_dec

    if args.include_only_first_k_sae_features:
        all_sae_features = all_sae_features[: args.include_only_first_k_sae_features]


    all_sims = all_sae_features @ all_sae_features.T

    # Set diagonal to 0
    all_sims.fill_diagonal_(0)

    if method == "graph":
        cutoffs = list(np.logspace(start=-1, stop=0, num=20)) + [0.5]
        cutoffs = [round(c, 2) for c in cutoffs]
        for top_k_for_graph in [2, 3, 4, 5, 6, 8, 16, 32]:
            for sim_cutoff in cutoffs:
                print("\ntopk", top_k_for_graph, "\t sim cutoff", sim_cutoff)
                components = graph_cluster_sims(all_sims, top_k_for_graph=top_k_for_graph, sim_cutoff=sim_cutoff)
                print("n_clusters", len(components))
                if 5 <= len(components) <= 200:
                    with open(
                        root_save_dir / f"{model_name}_layer_{layer}_clusters_cutoff_{sim_cutoff}_topk_{top_k_for_graph}.pkl", "wb"
                    ) as f:
                        pickle.dump(components, f)
                    print("saved.")
                else:
                    print("discard")

    else:
        all_sims = torch.clamp(all_sims, -1, 1)
        all_sims = 1 - torch.arccos(all_sims) / torch.pi
        all_sims = all_sims.detach().cpu().numpy()
        for n_clusters in [10, 25, 50, 100]:
            print("n_clusters", n_clusters)
            clusters = spectral_cluster_sims(all_sims, n_clusters)
            pickle.dump(
                clusters,
                open(root_save_dir / f"{model_name}_layer_{layer}_clusters_spectral_n{len(clusters)}.pkl", "wb"),
            )
            print("saved")

if method == "graph":
    paths = os.listdir(root_save_dir)
    cluster_configs = set([re.search(r"_clusters_(.+)\.pkl", p).group(1) for p in paths])
    layers = [5, 7, 9, 10]
    to_remove = set()
    for cluster_config in cluster_configs:
        if any([not (root_save_dir / f"gpt-2_layer_{layer}_clusters_{cluster_config}.pkl").exists() for layer in layers]):
            to_remove.add(cluster_config)

    for path in paths:
        if re.search(r"_clusters_(.+)\.pkl", path).group(1) in to_remove:
            os.remove(root_save_dir / path)
            print("remove", root_save_dir / path)
# rsync -avhu /scratch/xhuang/MultiDimensionalFeatures/sae_multid_feature_discovery/gpt-2_layer_9_clusters_spectral_n12.pkl  SIC:/home/xhuang/MultiDimensionalFeatures/sae_multid_feature_discovery/
