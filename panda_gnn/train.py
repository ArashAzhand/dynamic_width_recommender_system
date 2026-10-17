"""Objective Function & Training Harness Module for PANDA Collaborative Filtering."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from panda_gnn.data import DatasetSplits
from panda_gnn.evaluator import EvaluationResults, FullCatalogueEvaluator
from panda_gnn.models import PANDARecommender


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameter configuration for BPR model training."""

    learning_rate: float = 1e-3
    lambda_reg: float = 1e-4
    epochs: int = 300
    patience: int = 50
    batch_size: int = 2048
    eval_frequency: int = 5
    seed: int = 42


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) Loss with L2 Embedding Regularization."""

    def __init__(self, lambda_reg: float = 1e-4) -> None:
        """Initializes BPR loss function."""
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
        neg_emb0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes BPR loss and initial embedding regularization."""
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = (
            (1.0 / 2.0)
            * (
                u_emb0.norm(2).pow(2)
                + pos_emb0.norm(2).pow(2)
                + neg_emb0.norm(2).pow(2)
            )
            / pos_scores.size(0)
        )
        total_loss = bpr_loss + self.lambda_reg * reg_loss
        return total_loss, bpr_loss, reg_loss


class NegativeSampler:
    """Uniform negative sampling engine for implicit feedback bipartite graphs."""

    def __init__(
        self,
        num_items: int,
        user_train_dict: Dict[int, Set[int]],
        seed: int = 42,
    ) -> None:
        """Initializes sampler."""
        self.num_items = num_items
        self.user_train_dict = user_train_dict
        self.rng = np.random.RandomState(seed)

    def sample_negatives_batch(self, users: torch.Tensor) -> torch.Tensor:
        """Samples one negative item for each given user."""
        users_np = users.cpu().numpy()
        negatives = np.empty(len(users_np), dtype=np.int64)

        for idx, u in enumerate(users_np):
            seen = self.user_train_dict[u]
            j = self.rng.randint(0, self.num_items)
            while j in seen:
                j = self.rng.randint(0, self.num_items)
            negatives[idx] = j

        return torch.tensor(negatives, dtype=torch.long, device=users.device)


class BaseTrainer(ABC):
    """Abstract Template Method class managing the model training lifecycle."""

    def __init__(
        self,
        model: PANDARecommender,
        splits: DatasetSplits,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        """Initializes trainer."""
        self.model = model.to(device)
        self.splits = splits
        self.config = config
        self.device = device

        self.loss_fn = BPRLoss(lambda_reg=config.lambda_reg)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.evaluator = FullCatalogueEvaluator(k_list=(10, 20))
        self.sampler = NegativeSampler(
            num_items=splits.num_items,
            user_train_dict=splits.user_train_dict,
            seed=config.seed,
        )

        self.best_metric: float = -1.0
        self.best_epoch: int = -1
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @abstractmethod
    def train_epoch(self) -> float:
        """Runs a single training epoch across all training edges."""
        pass

    def evaluate_validation(self) -> EvaluationResults:
        """Evaluates model against validation ground-truth items with training masking."""
        return self.evaluator.evaluate(
            model=self.model,
            eval_user_dict=self.splits.user_val_dict,
            mask_user_dict=self.splits.user_train_dict,
            train_edge_index=self.splits.train_edge_index,
            device=self.device,
        )

    def evaluate_test(self) -> EvaluationResults:
        """Evaluates best restored model against test ground-truth with train+val masking."""
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        combined_mask: Dict[int, Set[int]] = {
            u: self.splits.user_train_dict[u] | self.splits.user_val_dict[u]
            for u in range(self.splits.num_users)
        }

        return self.evaluator.evaluate(
            model=self.model,
            eval_user_dict=self.splits.user_test_dict,
            mask_user_dict=combined_mask,
            train_edge_index=self.splits.train_edge_index,
            device=self.device,
        )

    def fit(self, verbose: bool = True) -> EvaluationResults:
        """Executes full training harness with early stopping tracking Recall@20."""
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch()

            if epoch % self.config.eval_frequency == 0 or epoch == 1:
                val_res = self.evaluate_validation()
                current_recall = val_res.get("Recall@20")

                if verbose:
                    ndcg = val_res.get("NDCG@20")
                    print(
                        f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | "
                        f"Val Recall@20: {current_recall:.4f} | Val NDCG@20: {ndcg:.4f}"
                    )

                if current_recall > self.best_metric:
                    self.best_metric = current_recall
                    self.best_epoch = epoch
                    self.best_state_dict = deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        if verbose:
                            print(
                                f"Early stopping triggered at epoch {epoch} "
                                f"(Best Epoch: {self.best_epoch}, Best Val Recall@20: {self.best_metric:.4f})"
                            )
                        break

        return self.evaluate_test()


class BPRTrainer(BaseTrainer):
    """Concrete trainer executing full-graph BPR loss optimization."""

    def train_epoch(self) -> float:
        """Executes one epoch of training with full-graph propagation and negative sampling."""
        self.model.train()
        train_edges = self.splits.train_user_item_edges
        u_all = train_edges[0].to(self.device)
        pos_all = train_edges[1].to(self.device)

        self.optimizer.zero_grad()

        train_edge_index_gpu = self.splits.train_edge_index.to(self.device)
        user_emb, item_emb = self.model(train_edge_index_gpu)

        neg_all = self.sampler.sample_negatives_batch(u_all)

        pos_scores = self.model.predict_pairs(u_all, pos_all, user_emb, item_emb)
        neg_scores = self.model.predict_pairs(u_all, neg_all, user_emb, item_emb)

        u_0, pos_0, neg_0 = self.model.get_initial_embeddings(u_all, pos_all, neg_all)

        total_loss, _, _ = self.loss_fn(pos_scores, neg_scores, u_0, pos_0, neg_0)
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
