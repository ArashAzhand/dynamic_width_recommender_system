"""Full-Catalogue Evaluation Engine for Collaborative Filtering."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from panda_gnn.models import PANDARecommender


@dataclass(frozen=True)
class EvaluationResults:
    """Structured container holding recommendation ranking metrics across cutoffs."""

    metrics: Dict[str, float]

    def get(self, metric_name: str, default: float = 0.0) -> float:
        """Retrieves metric score by key."""
        return self.metrics.get(metric_name, default)

    def summary(self) -> str:
        """Generates formatted string summary of evaluated metrics."""
        lines = []
        for k in sorted(self.metrics.keys()):
            lines.append(f"{k}: {self.metrics[k]:.4f}")
        return " | ".join(lines)


class FullCatalogueEvaluator:
    """Vectorized full-catalogue evaluation engine."""

    def __init__(self, k_list: Tuple[int, ...] = (10, 20), batch_size: int = 512) -> None:
        """Initializes evaluator with evaluation cutoffs and batch size."""
        self.k_list = sorted(k_list)
        self.max_k = max(self.k_list)
        self.batch_size = batch_size

        ranks = np.arange(1, self.max_k + 1)
        self.discounts = 1.0 / np.log2(ranks + 1)

    def evaluate(
        self,
        model: PANDARecommender,
        eval_user_dict: Dict[int, Set[int]],
        mask_user_dict: Dict[int, Set[int]],
        train_edge_index: torch.Tensor,
        device: torch.device,
    ) -> EvaluationResults:
        """Runs full-catalogue evaluation across active users in eval set."""
        model.eval()

        eval_users = [u for u, items in eval_user_dict.items() if len(items) > 0]
        if not eval_users:
            return EvaluationResults(metrics={})

        with torch.no_grad():
            user_emb, item_emb = model(train_edge_index.to(device))

        metric_sums: Dict[str, float] = {
            f"{m}@{k}": 0.0
            for k in self.k_list
            for m in ["Recall", "NDCG", "Precision", "HitRate", "MAP", "MRR"]
        }
        total_eval_users = len(eval_users)

        for start_idx in range(0, total_eval_users, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_eval_users)
            batch_users = eval_users[start_idx:end_idx]

            u_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            scores = model.predict_all_items(u_tensor, user_emb, item_emb)

            for row_idx, u in enumerate(batch_users):
                seen_items = mask_user_dict.get(u, set())
                if seen_items:
                    seen_indices = torch.tensor(
                        list(seen_items), dtype=torch.long, device=device
                    )
                    scores[row_idx, seen_indices] = -float("inf")

            _, top_k_indices = torch.topk(scores, self.max_k, dim=-1)
            top_k_indices_np = top_k_indices.cpu().numpy()

            for row_idx, u in enumerate(batch_users):
                target_items = eval_user_dict[u]
                num_pos = len(target_items)
                recs = top_k_indices_np[row_idx]

                hits = np.array([1 if item in target_items else 0 for item in recs], dtype=np.float32)

                for k in self.k_list:
                    k_hits = hits[:k]
                    num_hits = k_hits.sum()

                    metric_sums[f"Recall@{k}"] += num_hits / num_pos
                    metric_sums[f"Precision@{k}"] += num_hits / k
                    metric_sums[f"HitRate@{k}"] += 1.0 if num_hits > 0 else 0.0

                    dcg = (k_hits * self.discounts[:k]).sum()
                    idcg_len = min(k, num_pos)
                    idcg = self.discounts[:idcg_len].sum()
                    metric_sums[f"NDCG@{k}"] += (dcg / idcg) if idcg > 0 else 0.0

                    hit_positions = np.where(k_hits == 1)[0]
                    if len(hit_positions) > 0:
                        first_hit_rank = hit_positions[0] + 1
                        metric_sums[f"MRR@{k}"] += 1.0 / first_hit_rank

                    if num_hits > 0:
                        precisions = np.cumsum(k_hits) / np.arange(1, k + 1)
                        ap = (precisions * k_hits).sum() / min(k, num_pos)
                        metric_sums[f"MAP@{k}"] += ap

        averaged_metrics = {k: v / total_eval_users for k, v in metric_sums.items()}
        return EvaluationResults(metrics=averaged_metrics)
