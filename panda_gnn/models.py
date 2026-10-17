"""Neural Architecture & Prediction Alignment Module for PANDA Graph Recommenders."""

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from panda_gnn.conv_layers import (
    BasePANDAConv,
    PANDAGATConv,
    PANDAGCNConv,
    PANDAGINConv,
    PANDALightGCNConv,
    PANDASAGEConv,
    apply_mask,
)


class BackboneType(str, Enum):
    """Supported graph neural network convolution backbones."""

    LIGHTGCN = "LIGHTGCN"
    GCN = "GCN"
    GRAPHSAGE = "GRAPHSAGE"
    GAT = "GAT"
    GIN = "GIN"


class ConvKernelFactory:
    """Factory creating width-aware convolutional kernels according to backbone type."""

    @staticmethod
    def create_kernel(
        backbone_type: BackboneType,
        d_low: int,
        d_high: int,
        gat_heads: int = 2,
    ) -> BasePANDAConv:
        """Instantiates corresponding PANDA convolution kernel."""
        if backbone_type == BackboneType.LIGHTGCN:
            return PANDALightGCNConv(d_low=d_low, d_high=d_high)
        elif backbone_type == BackboneType.GCN:
            return PANDAGCNConv(d_low=d_low, d_high=d_high)
        elif backbone_type == BackboneType.GRAPHSAGE:
            return PANDASAGEConv(d_low=d_low, d_high=d_high)
        elif backbone_type == BackboneType.GAT:
            return PANDAGATConv(d_low=d_low, d_high=d_high, heads=gat_heads)
        elif backbone_type == BackboneType.GIN:
            return PANDAGINConv(d_low=d_low, d_high=d_high)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")


class PredictionAlignmentLayer(nn.Module):
    """Maps high-dimensional hub embeddings down to d_low after message passing."""

    def __init__(self, d_low: int, d_high: int) -> None:
        """Initializes alignment projection layer."""
        super().__init__()
        self.d_low = d_low
        self.d_high = d_high
        self.W_align = nn.Linear(d_high, d_low, bias=False)
        nn.init.xavier_normal_(self.W_align.weight)

    def forward(self, x: torch.Tensor, is_high: torch.Tensor) -> torch.Tensor:
        """Projects high-width embeddings down to d_low, retaining low-width intact."""
        out = torch.empty((x.size(0), self.d_low), device=x.device, dtype=x.dtype)
        low_mask = ~is_high

        if low_mask.any():
            out[low_mask] = x[low_mask, : self.d_low]

        if is_high.any():
            out[is_high] = self.W_align(x[is_high])

        return out


class PANDARecommender(nn.Module):
    """Unified PANDA Graph Recommender Architecture."""

    def __init__(
        self,
        backbone_type: BackboneType,
        num_users: int,
        num_items: int,
        d_low: int = 64,
        d_high: int = 128,
        num_layers: int = 3,
        selection_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gat_heads: int = 2,
    ) -> None:
        """Initializes PANDARecommender."""
        super().__init__()
        self.backbone_type = backbone_type
        self.num_users = num_users
        self.num_items = num_items
        self.d_low = d_low
        self.d_high = d_high
        self.num_layers = num_layers

        if selection_masks is not None:
            user_mask, item_mask = selection_masks
            self.register_buffer("user_mask", user_mask.bool())
            self.register_buffer("item_mask", item_mask.bool())
            is_high = torch.cat([user_mask, item_mask], dim=0).bool()
        else:
            self.register_buffer("user_mask", torch.zeros(num_users, dtype=torch.bool))
            self.register_buffer("item_mask", torch.zeros(num_items, dtype=torch.bool))
            is_high = torch.zeros(num_users + num_items, dtype=torch.bool)

        self.register_buffer("is_high", is_high)

        self.user_embedding = nn.Embedding(num_users, d_high)
        self.item_embedding = nn.Embedding(num_items, d_high)
        self._init_embeddings()

        self.conv_layers = nn.ModuleList(
            [
                ConvKernelFactory.create_kernel(
                    backbone_type=backbone_type,
                    d_low=d_low,
                    d_high=d_high,
                    gat_heads=gat_heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.align_layer = PredictionAlignmentLayer(d_low=d_low, d_high=d_high)

    def _init_embeddings(self) -> None:
        """Initializes node embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(
        self, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs multi-layer message passing and projects to aligned d_low space."""
        e_u0 = self.user_embedding.weight
        e_i0 = self.item_embedding.weight
        x0 = torch.cat([e_u0, e_i0], dim=0)
        x = apply_mask(x0, self.is_high, self.d_low)

        all_layer_reps = [x]

        for conv in self.conv_layers:
            x = conv(x, edge_index, self.is_high)
            all_layer_reps.append(x)

        if self.backbone_type == BackboneType.LIGHTGCN:
            stacked = torch.stack(all_layer_reps, dim=0)
            final_rep = torch.mean(stacked, dim=0)
        else:
            final_rep = all_layer_reps[-1]

        aligned = self.align_layer(final_rep, self.is_high)

        user_aligned = aligned[: self.num_users]
        item_aligned = aligned[self.num_users :]

        return user_aligned, item_aligned

    def get_initial_embeddings(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves raw 0-th layer initial embeddings for L2 regularization."""
        u_emb0 = self.user_embedding(users)
        pos_emb0 = self.item_embedding(pos_items)
        neg_emb0 = self.item_embedding(neg_items)
        return u_emb0, pos_emb0, neg_emb0

    def predict_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Computes inner product prediction for pairwise user-item tuples."""
        u_e = user_emb[users]
        i_e = item_emb[items]
        return (u_e * i_e).sum(dim=-1)

    def predict_all_items(
        self,
        user_indices: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Computes full affinity score matrix for given users against all catalogue items."""
        u_e = user_emb[user_indices]
        return torch.matmul(u_e, item_emb.t())
