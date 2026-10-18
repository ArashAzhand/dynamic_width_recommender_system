"""Master MovieLens-1M Benchmark Runner.

Executes all 5 GNN backbones across all 6 capacity configurations on MovieLens-1M:
  1. LIGHTGCN (Vanilla, Param-Matched d=73, PANDA-Random, PANDA-Degree, PANDA-PageRank, PANDA-Betweenness)
  2. GCN (Vanilla, Param-Matched d=73, PANDA-Random, PANDA-Degree, PANDA-PageRank, PANDA-Betweenness)
  3. GRAPHSAGE (Vanilla, Param-Matched d=73, PANDA-Random, PANDA-Degree, PANDA-PageRank, PANDA-Betweenness)
  4. GAT (Vanilla, Param-Matched d=73, PANDA-Random, PANDA-Degree, PANDA-PageRank, PANDA-Betweenness)
  5. GIN (Vanilla, Param-Matched d=73, PANDA-Random, PANDA-Degree, PANDA-PageRank, PANDA-Betweenness)

Saves results incrementally to benchmark_results_ml1m.csv and synchronizes Report/main.pdf.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from panda_gnn.data import DatasetFactory, DatasetSplits
from panda_gnn.models import BackboneType, PANDARecommender
from panda_gnn.selectors import (
    BetweennessCentralitySelector,
    DegreeCentralitySelector,
    PageRankSelector,
    RandomUniformSelector,
)
from panda_gnn.train import BPRTrainer, TrainingConfig


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_param_matched_dimension(num_users: int, num_items: int, d_low: int = 64, d_high: int = 128, k: float = 0.10) -> int:
    total_nodes = num_users + num_items
    num_high = int(round(total_nodes * k))
    num_low = total_nodes - num_high
    total_params = (num_low * d_low) + (num_high * d_high) + (3 * d_low * d_high)
    return int(round(total_params / total_nodes))


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting MovieLens-1M Full Benchmark Suite on {device} ===")

    seed = 42
    set_seed(seed)
    loader = DatasetFactory.get_loader("ML-1M", seed=seed)
    splits = loader.load_and_split()

    num_users = splits.num_users
    num_items = splits.num_items
    d_low = 64
    d_high = 128
    k_percentile = 0.10
    d_matched = compute_param_matched_dimension(num_users, num_items, d_low, d_high, k_percentile)

    print(f"Loaded MovieLens-1M: {num_users} Users, {num_items} Items, {splits.train_user_item_edges.size(1)} Train Edges")
    print(f"Base d_L={d_low}, Expanded d_H={d_high}, Param-Matched d={d_matched}")

    # Precompute selection masks once so they are reused across all 5 backbones
    print("\n--- Precomputing Topological Selection Masks ---")
    t0 = time.time()
    rand_sel = RandomUniformSelector(k_percentile=k_percentile, seed=seed)
    random_mask = rand_sel.select(splits.train_edge_index, num_users, num_items)
    print(f"1. Random Mask ready (took {time.time()-t0:.2f}s)")

    t0 = time.time()
    deg_sel = DegreeCentralitySelector(k_percentile=k_percentile, seed=seed)
    degree_mask = deg_sel.select(splits.train_edge_index, num_users, num_items)
    print(f"2. Degree Mask ready (took {time.time()-t0:.2f}s)")

    t0 = time.time()
    pr_sel = PageRankSelector(k_percentile=k_percentile, seed=seed)
    pagerank_mask = pr_sel.select(splits.train_edge_index, num_users, num_items)
    print(f"3. PageRank Mask ready (took {time.time()-t0:.2f}s)")

    t0 = time.time()
    bet_sel = BetweennessCentralitySelector(k_percentile=k_percentile, seed=seed, sample_k=50)
    betweenness_mask = bet_sel.select(splits.train_edge_index, num_users, num_items)
    print(f"4. Betweenness Mask ready (took {time.time()-t0:.2f}s)")

    configs = [
        ("Vanilla (d=64)", None, d_low, d_low, True),
        (f"Param-Matched (d={d_matched})", None, d_matched, d_matched, True),
        ("PANDA-Random", random_mask, d_low, d_high, False),
        ("PANDA-Degree", degree_mask, d_low, d_high, False),
        ("PANDA-PageRank", pagerank_mask, d_low, d_high, False),
        ("PANDA-Betweenness", betweenness_mask, d_low, d_high, False),
    ]

    backbones = [
        BackboneType.LIGHTGCN,
        BackboneType.GCN,
        BackboneType.GRAPHSAGE,
        BackboneType.GAT,
        BackboneType.GIN,
    ]

    csv_path = "benchmark_results_ml1m.csv"
    existing_results: Dict[str, Dict[str, str]] = {}
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            for _, row in df_existing.iterrows():
                cfg_key = row["Model Configuration"]
                existing_results[cfg_key] = row.to_dict()
        except Exception as e:
            print(f"Could not load existing CSV: {e}")

    total_runs = len(backbones) * len(configs)
    current_idx = 0

    for backbone in backbones:
        print(f"\n=======================================================")
        print(f" EVALUATING BACKBONE: {backbone.value}")
        print(f"=======================================================")

        for config_name, mask_pair, d_l, d_h, is_vanilla in configs:
            current_idx += 1
            model_key = f"{backbone.value} - {config_name}"

            # If already evaluated with 5-fold cross-validation or previous run, we can check
            # For LightGCN, if existing results has standard format and already run, keep or re-run
            # If we want all backbones cleanly present:
            if model_key in existing_results and backbone == BackboneType.LIGHTGCN:
                print(f"[{current_idx}/{total_runs}] {model_key} already recorded, retaining.")
                continue

            print(f"[{current_idx}/{total_runs}] Running {model_key}...", end="", flush=True)
            set_seed(seed)

            eff_d_high = d_l if is_vanilla else d_h
            model = PANDARecommender(
                backbone_type=backbone,
                num_users=num_users,
                num_items=num_items,
                d_low=d_l,
                d_high=eff_d_high,
                num_layers=3,
                selection_masks=mask_pair if not is_vanilla else None,
            )

            # Patience: 20 checks = 100 epochs ceiling 300 epochs
            cfg = TrainingConfig(
                learning_rate=1e-3,
                lambda_reg=1e-4,
                epochs=300,
                patience=20,
                eval_frequency=5,
                seed=seed,
            )

            trainer = BPRTrainer(model=model, splits=splits, config=cfg, device=device)
            t_train_start = time.time()
            test_results = trainer.fit(verbose=False)
            elapsed = time.time() - t_train_start

            metrics = test_results.metrics
            rec20 = metrics.get("Recall@20", 0.0)
            ndcg20 = metrics.get("NDCG@20", 0.0)
            print(
                f" Done in {elapsed:.1f}s! (Best Epoch: {trainer.best_epoch}) -> "
                f"Recall@20: {rec20:.4f}, NDCG@20: {ndcg20:.4f}"
            )

            # Store in existing_results dictionary
            existing_results[model_key] = {
                "Dataset": "ML-1M",
                "Backbone": backbone.value,
                "Model Configuration": model_key,
                "Recall@10": f"{metrics.get('Recall@10', 0.0):.4f}",
                "Recall@20": f"{metrics.get('Recall@20', 0.0):.4f}",
                "NDCG@10": f"{metrics.get('NDCG@10', 0.0):.4f}",
                "NDCG@20": f"{metrics.get('NDCG@20', 0.0):.4f}",
                "MAP@20": f"{metrics.get('MAP@20', 0.0):.4f}",
                "MRR@20": f"{metrics.get('MRR@20', 0.0):.4f}",
                "Best_Epoch": f"{trainer.best_epoch:.1f}",
            }

            # Save incrementally after each configuration completes
            df_out = pd.DataFrame(list(existing_results.values()))
            df_out.to_csv(csv_path, index=False)

        # Sync LaTeX report after each backbone finishes
        try:
            from Report.sync_report import sync_all
            print(f"\n[Checkpoint] Synchronizing Report/main.pdf after {backbone.value}...")
            sync_all(report_dir="Report")
        except Exception as e:
            print(f"Sync error: {e}")

    print("\n=======================================================")
    print(" ALL 30 EXPERIMENTS ON MOVIELENS-1M COMPLETE!")
    print("=======================================================")
    final_df = pd.DataFrame(list(existing_results.values()))
    print(final_df.to_string(index=False))
    final_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    run_benchmark()
