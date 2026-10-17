"""Node Selection Engine Module for PANDA Width-Aware Architectures."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


class NodeSelectionStrategy(ABC):
    """Abstract base class defining the strategy interface for node selection."""

    def __init__(self, k_percentile: float = 0.10, seed: int = 42) -> None:
        """Initializes selector parameters."""
        if not 0.0 <= k_percentile <= 1.0:
            raise ValueError(f"k_percentile must be in [0.0, 1.0], got {k_percentile}")
        self.k_percentile = k_percentile
        self.seed = seed

    @abstractmethod
    def select(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates centrality and returns boolean masks for users and items."""
        pass

    @staticmethod
    def _extract_bipartite_edges(
        edge_index: torch.Tensor, num_users: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes edge_index to unshifted user and item indices."""
        src = edge_index[0]
        dst = edge_index[1]

        if dst.max() >= num_users:
            u_mask = (src < num_users) & (dst >= num_users)
            users = src[u_mask]
            items = dst[u_mask] - num_users
        else:
            users = src
            items = dst

        return users, items


class DegreeCentralitySelector(NodeSelectionStrategy):
    """Selects top-k% highest-degree nodes based on interaction count."""

    def select(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects top-k% nodes according to degree centrality."""
        users, items = self._extract_bipartite_edges(edge_index, num_users)

        user_degrees = torch.bincount(users, minlength=num_users).float()
        item_degrees = torch.bincount(items, minlength=num_items).float()

        q = 1.0 - self.k_percentile
        user_cutoff = torch.quantile(user_degrees, q)
        item_cutoff = torch.quantile(item_degrees, q)

        user_mask = user_degrees >= user_cutoff
        item_mask = item_degrees >= item_cutoff

        return user_mask, item_mask


class BetweennessCentralitySelector(NodeSelectionStrategy):
    """Selects top-k% nodes based on shortest-path Betweenness Centrality."""

    def __init__(
        self,
        k_percentile: float = 0.10,
        seed: int = 42,
        sample_k: Optional[int] = None,
    ) -> None:
        """Initializes Betweenness Centrality selector."""
        super().__init__(k_percentile=k_percentile, seed=seed)
        self.sample_k = sample_k

    def select(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Betweenness Centrality on the bipartite graph."""
        users, items = self._extract_bipartite_edges(edge_index, num_users)

        graph = nx.Graph()
        graph.add_nodes_from(range(num_users), bipartite=0)
        graph.add_nodes_from(range(num_users, num_users + num_items), bipartite=1)

        edges = list(zip(users.cpu().numpy(), (items + num_users).cpu().numpy()))
        graph.add_edges_from(edges)

        bc = nx.betweenness_centrality(
            graph,
            k=self.sample_k,
            seed=self.seed,
            normalized=True,
        )

        user_scores = torch.tensor([bc[u] for u in range(num_users)], dtype=torch.float32)
        item_scores = torch.tensor(
            [bc[num_users + i] for i in range(num_items)], dtype=torch.float32
        )

        q = 1.0 - self.k_percentile
        user_cutoff = torch.quantile(user_scores, q)
        item_cutoff = torch.quantile(item_scores, q)

        user_mask = user_scores >= user_cutoff
        item_mask = item_scores >= item_cutoff

        return user_mask, item_mask


class PageRankSelector(NodeSelectionStrategy):
    """Selects top-k% nodes based on Bipartite PageRank stationary distribution."""

    def __init__(
        self,
        k_percentile: float = 0.10,
        alpha: float = 0.85,
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        """Initializes PageRank selector."""
        super().__init__(k_percentile=k_percentile, seed=seed)
        self.alpha = alpha
        self.max_iter = max_iter

    def select(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes PageRank on bipartite graph."""
        users, items = self._extract_bipartite_edges(edge_index, num_users)

        graph = nx.Graph()
        graph.add_nodes_from(range(num_users + num_items))
        edges = list(zip(users.cpu().numpy(), (items + num_users).cpu().numpy()))
        graph.add_edges_from(edges)

        pr = nx.pagerank(graph, alpha=self.alpha, max_iter=self.max_iter)

        user_scores = torch.tensor([pr[u] for u in range(num_users)], dtype=torch.float32)
        item_scores = torch.tensor(
            [pr[num_users + i] for i in range(num_items)], dtype=torch.float32
        )

        q = 1.0 - self.k_percentile
        user_cutoff = torch.quantile(user_scores, q)
        item_cutoff = torch.quantile(item_scores, q)

        user_mask = user_scores >= user_cutoff
        item_mask = item_scores >= item_cutoff

        return user_mask, item_mask


class RandomUniformSelector(NodeSelectionStrategy):
    """Ablation strategy uniformly sampling k% of nodes at random."""

    def select(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly marks k% of users and k% of items as high-capacity."""
        rng = np.random.RandomState(self.seed)

        num_high_users = int(round(num_users * self.k_percentile))
        num_high_items = int(round(num_items * self.k_percentile))

        user_perm = rng.permutation(num_users)
        item_perm = rng.permutation(num_items)

        user_mask = torch.zeros(num_users, dtype=torch.bool)
        item_mask = torch.zeros(num_items, dtype=torch.bool)

        user_mask[user_perm[:num_high_users]] = True
        item_mask[item_perm[:num_high_items]] = True

        return user_mask, item_mask


def plot_centrality_cutoff(
    scores: torch.Tensor,
    cutoff: float,
    node_type: str = "User",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plots centrality score distribution with threshold cutoff line."""
    scores_np = scores.cpu().numpy()
    plt.figure(figsize=(8, 4.5))
    plt.hist(scores_np, bins=40, color="#3498db", edgecolor="black", alpha=0.7)
    plt.axvline(
        cutoff,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Cutoff = {cutoff:.3f}",
    )
    plt.title(title or f"{node_type} Centrality Distribution & Hub Cutoff")
    plt.xlabel("Centrality Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
