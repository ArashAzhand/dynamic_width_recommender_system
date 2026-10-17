"""Benchmark Pipeline Runner Module for PANDA Collaborative Filtering.

Orchestrates multi-seed benchmarking across GNN backbones and node selection strategies:
  1. Vanilla Baselines (uniform width d_L = 64)
  2. Parameter-Matched Baselines (uniform width d_matched to match total PANDA capacity)
  3. Ablation PANDA (RandomUniformSelector)
  4. Proposed PANDA (DegreeCentrality, BetweennessCentrality, PageRank)

Reports Mean +/- Std Dev across random seeds for all metrics.
"""

import argparse
from collections import defaultdict
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from panda_gnn.data import DatasetFactory, DatasetSplits, MovieLens100KLoader
from panda_gnn.models import BackboneType, PANDARecommender
from panda_gnn.selectors import (
    BetweennessCentralitySelector,
    DegreeCentralitySelector,
    NodeSelectionStrategy,
    PageRankSelector,
    RandomUniformSelector,
    plot_centrality_cutoff,
)
from panda_gnn.train import BPRTrainer, TrainingConfig


def set_seed(seed: int) -> None:
    """Sets global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_param_matched_dimension(
    num_users: int,
    num_items: int,
    d_low: int = 64,
    d_high: int = 128,
    k_percentile: float = 0.10,
) -> int:
    """Calculates equivalent uniform width that matches total PANDA embedding parameters.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        d_low: Base low embedding dimension.
        d_high: Expanded high embedding dimension.
        k_percentile: Fraction of nodes allocated high dimension.

    Returns:
        Integer uniform embedding dimension matching PANDA capacity.
    """
    total_nodes = num_users + num_items
    num_high = int(round(total_nodes * k_percentile))
    num_low = total_nodes - num_high

    # Embedding params + routing projection matrices (W_f, W_g, W_align)
    total_panda_params = (num_low * d_low) + (num_high * d_high) + (3 * d_low * d_high)
    d_matched = int(round(total_panda_params / total_nodes))
    return d_matched


def run_single_experiment(
    backbone: BackboneType,
    splits: DatasetSplits,
    selector: Optional[NodeSelectionStrategy],
    d_low: int,
    d_high: int,
    config: TrainingConfig,
    device: torch.device,
    is_vanilla: bool = False,
    precomputed_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Runs a single training and evaluation cycle for given configuration.

    Args:
        backbone: Chosen GNN backbone.
        splits: Dataset splits and graph structures.
        selector: Node selection strategy (None for vanilla baselines).
        d_low: Low embedding dimension.
        d_high: High embedding dimension.
        config: Training configuration.
        device: Torch compute device.
        is_vanilla: If True, disables PANDA routing (uniform dimension).
        precomputed_masks: Optional precomputed (user_mask, item_mask).

    Returns:
        Dictionary of test metric scores.
    """
    num_users = splits.num_users
    num_items = splits.num_items

    if is_vanilla or (selector is None and precomputed_masks is None):
        # Uniform dimension baseline (all nodes allocated d_low, no routing)
        selection_masks = None
        eff_d_high = d_low
    elif precomputed_masks is not None:
        selection_masks = precomputed_masks
        eff_d_high = d_high
    else:
        user_mask, item_mask = selector.select(
            splits.train_edge_index, num_users, num_items
        )
        selection_masks = (user_mask, item_mask)
        eff_d_high = d_high

    model = PANDARecommender(
        backbone_type=backbone,
        num_users=num_users,
        num_items=num_items,
        d_low=d_low,
        d_high=eff_d_high,
        num_layers=3,
        selection_masks=selection_masks,
    )

    trainer = BPRTrainer(
        model=model,
        splits=splits,
        config=config,
        device=device,
    )

    test_results = trainer.fit(verbose=False)
    return test_results.metrics, trainer.best_epoch, trainer.best_metric


def main() -> None:
    """Main benchmark execution harness."""
    parser = argparse.ArgumentParser(description="PANDA Recommendation Benchmark Runner")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
        help="Random seeds for benchmark runs",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=1,
        help="Number of folds for K-fold cross-validation (e.g. 5 for 5-fold CV). If > 1, evaluates across folds.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ML-100K",
        help="Benchmark dataset: ML-100K or ML-1M",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=["LIGHTGCN", "GCN", "GRAPHSAGE", "GAT", "GIN"],
        help="GNN Backbones to evaluate",
    )
    parser.add_argument("--epochs", type=int, default=400, help="Training epoch ceiling for early stopping")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (eval checks, 5 epochs each, default 50 checks = 250 epochs)")
    parser.add_argument("--d_low", type=int, default=64, help="Base dimension")
    parser.add_argument("--d_high", type=int, default=128, help="Hub dimension")
    parser.add_argument("--k_percentile", type=float, default=0.10, help="Hub percentile")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_reg", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--fast_run", action="store_true", help="Quick sanity run (1 seed/fold, 25 epochs)")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Output path")
    parser.add_argument("--plot_centrality", action="store_true", help="Generate centrality plots")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Benchmark on device: {device}")

    seeds = [42] if args.fast_run else args.seeds
    epochs = 25 if args.fast_run else args.epochs

    # Map backbones
    selected_backbones = [BackboneType(b.upper()) for b in args.backbones]

    # Pre-load data once for profiling
    data_loader = DatasetFactory.get_loader(args.dataset, seed=seeds[0])
    sample_splits = data_loader.load_and_split()
    num_users = sample_splits.num_users
    num_items = sample_splits.num_items

    d_matched = compute_param_matched_dimension(
        num_users, num_items, args.d_low, args.d_high, args.k_percentile
    )
    print(
        f"Graph Info: {num_users} Users, {num_items} Items | "
        f"Base d_L={args.d_low}, Expanded d_H={args.d_high}, Param-Matched d={d_matched}"
    )

    if args.plot_centrality:
        os.makedirs("plots", exist_ok=True)
        users, items = DegreeCentralitySelector._extract_bipartite_edges(
            sample_splits.train_edge_index, num_users
        )
        u_deg = torch.bincount(users, minlength=num_users).float()
        i_deg = torch.bincount(items, minlength=num_items).float()
        u_cutoff = float(torch.quantile(u_deg, 1.0 - args.k_percentile))
        i_cutoff = float(torch.quantile(i_deg, 1.0 - args.k_percentile))
        plot_centrality_cutoff(
            u_deg, u_cutoff, "User", "User Degree Centrality", "plots/user_degree_cutoff.png"
        )
        plot_centrality_cutoff(
            i_deg, i_cutoff, "Item", "Item Degree Centrality", "plots/item_degree_cutoff.png"
        )
        print("Generated centrality cutoff visualization plots in plots/")

    # Setup experimental configurations
    results_collector: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    configs_to_test = [
        ("Vanilla (d=64)", None, args.d_low, args.d_low, True),
        ("Param-Matched (d=80)", None, d_matched, d_matched, True),
        (
            "PANDA-Random",
            lambda s: RandomUniformSelector(k_percentile=args.k_percentile, seed=s),
            args.d_low,
            args.d_high,
            False,
        ),
        (
            "PANDA-Degree",
            lambda s: DegreeCentralitySelector(k_percentile=args.k_percentile, seed=s),
            args.d_low,
            args.d_high,
            False,
        ),
        (
            "PANDA-PageRank",
            lambda s: PageRankSelector(k_percentile=args.k_percentile, seed=s),
            args.d_low,
            args.d_high,
            False,
        ),
        (
            "PANDA-Betweenness",
            lambda s: BetweennessCentralitySelector(
                k_percentile=args.k_percentile,
                seed=s,
                sample_k=50 if "1M" in args.dataset.upper() else 300,
            ),
            args.d_low,
            args.d_high,
            False,
        ),
    ]

    if args.num_folds > 1:
        splits_units = [("Fold", f) for f in range(args.num_folds)]
    else:
        splits_units = [("Seed", s) for s in seeds]

    total_runs = len(selected_backbones) * len(configs_to_test) * len(splits_units)
    run_idx = 0
    mask_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    for backbone in selected_backbones:
        print(f"\n=======================================================")
        print(f" BENCHMARKING BACKBONE: {backbone.value}")
        print(f"=======================================================")

        for config_name, selector_factory, d_l, d_h, is_vanilla in configs_to_test:
            model_key = f"{backbone.value} - {config_name}"

            for unit_type, unit_val in splits_units:
                run_idx += 1
                seed = seeds[0] if unit_type == "Fold" else unit_val
                set_seed(seed)
                loader = DatasetFactory.get_loader(args.dataset, seed=seed)

                if unit_type == "Fold":
                    splits = loader.load_kfold_splits(num_folds=args.num_folds, fold_idx=unit_val)
                    run_label = f"Fold {unit_val + 1}/{args.num_folds}"
                else:
                    splits = loader.load_and_split()
                    run_label = f"Seed {unit_val}"

                train_cfg = TrainingConfig(
                    learning_rate=args.lr,
                    lambda_reg=args.lambda_reg,
                    epochs=epochs,
                    patience=args.patience,
                    eval_frequency=5,
                    seed=seed,
                )

                # Cache selection masks so PageRank / Betweenness are computed only once per fold/seed
                cache_key = (config_name, unit_type, unit_val)
                if cache_key in mask_cache:
                    cached_masks = mask_cache[cache_key]
                elif selector_factory is not None and not is_vanilla:
                    selector = selector_factory(seed)
                    cached_masks = selector.select(
                        splits.train_edge_index, splits.num_users, splits.num_items
                    )
                    mask_cache[cache_key] = cached_masks
                else:
                    cached_masks = None

                print(
                    f"[{run_idx}/{total_runs}] Running {model_key} ({run_label})...",
                    end="",
                    flush=True,
                )
                metrics, best_epoch, best_metric = run_single_experiment(
                    backbone=backbone,
                    splits=splits,
                    selector=None,
                    d_low=d_l,
                    d_high=d_h,
                    config=train_cfg,
                    device=device,
                    is_vanilla=is_vanilla,
                    precomputed_masks=cached_masks,
                )

                rec20 = metrics.get("Recall@20", 0.0)
                ndcg20 = metrics.get("NDCG@20", 0.0)
                print(
                    f" Done! (Best Epoch: {best_epoch}, Val R@20: {best_metric:.4f}) -> "
                    f"Recall@20: {rec20:.4f}, NDCG@20: {ndcg20:.4f}"
                )

                for m_name, val in metrics.items():
                    results_collector[model_key][m_name].append(val)
                results_collector[model_key]["Best_Epoch"].append(float(best_epoch))

    # Compile Summary Table (Mean +/- Std)
    summary_rows = []
    key_metrics = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@20", "MRR@20", "Best_Epoch"]

    for model_key, metric_dict in results_collector.items():
        row = {"Model Configuration": model_key}
        for m in key_metrics:
            vals = metric_dict.get(m, [])
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                if m == "Best_Epoch":
                    row[m] = f"{mean_val:.1f} +/- {std_val:.1f}"
                else:
                    row[m] = f"{mean_val:.4f} +/- {std_val:.4f}"
            else:
                row[m] = "N/A"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\n\n=======================================================")
    print(f" FINAL BENCHMARK RESULTS (Mean +/- Std across {splits_units[0][0].lower()}s)")
    print("=======================================================")
    print(df_summary.to_string(index=False))

    df_summary.to_csv(args.output_csv, index=False)
    print(f"\nResults saved successfully to {args.output_csv}")

    # Synchronize LaTeX Report & PDF automatically
    try:
        from Report.sync_report import sync_all
        print("\nSynchronizing LaTeX report and compiling PDF in Report/...")
        sync_all(csv_path=args.output_csv, report_dir="Report")
    except Exception as e:
        print(f"Note: Could not automatically sync Report: {e}")


if __name__ == "__main__":
    main()
