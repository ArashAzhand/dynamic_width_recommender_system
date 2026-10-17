"""Width-Aware Convolutional Kernels for PANDA Graph Recommenders."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


def apply_mask(x: torch.Tensor, is_high: torch.Tensor, d_low: int) -> torch.Tensor:
    """Enforces zero-padding on trailing dimensions for low-width nodes out-of-place."""
    if is_high.all():
        return x
    mask_2d = is_high.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
    low_part = x[:, :d_low]
    high_part = x[:, d_low:] * mask_2d
    return torch.cat([low_part, high_part], dim=-1)


class BasePANDAConv(MessagePassing, ABC):
    """Abstract base convolution kernel providing width-aware dimension routing."""

    def __init__(self, d_low: int, d_high: int, aggr: str = "add", **kwargs) -> None:
        """Initializes base width routing projections."""
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        self.d_low = d_low
        self.d_high = d_high

        self.W_f = nn.Linear(d_low, d_high, bias=False)
        self.W_g = nn.Linear(d_high, d_low, bias=False)

        self._reset_routing_parameters()

    def _reset_routing_parameters(self) -> None:
        """Initializes projection matrices with Xavier normal."""
        nn.init.xavier_normal_(self.W_f.weight)
        nn.init.xavier_normal_(self.W_g.weight)

    def route_messages(
        self,
        x_j: torch.Tensor,
        is_high_i: torch.Tensor,
        is_high_j: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized PANDA dimension routing for source messages."""
        device = x_j.device

        low_to_high = (~is_high_j) & is_high_i
        high_to_low = is_high_j & (~is_high_i)

        msg = x_j.clone()

        if low_to_high.any():
            low_feats = x_j[low_to_high, : self.d_low]
            msg[low_to_high] = self.W_f(low_feats)

        if high_to_low.any():
            high_feats = x_j[high_to_low]
            proj_low = self.W_g(high_feats)
            zeros = torch.zeros(
                proj_low.size(0), self.d_high - self.d_low, device=device, dtype=x_j.dtype
            )
            msg[high_to_low] = torch.cat([proj_low, zeros], dim=-1)

        return msg

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_high: torch.Tensor,
    ) -> torch.Tensor:
        """Forward convolution pass across graph."""
        pass


class PANDALightGCNConv(BasePANDAConv):
    """PANDA LightGCN Convolution: Width routing + Symmetric normalized sum."""

    def __init__(self, d_low: int, d_high: int) -> None:
        """Initializes PANDALightGCNConv with sum aggregation."""
        super().__init__(d_low=d_low, d_high=d_high, aggr="add")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_high: torch.Tensor,
    ) -> torch.Tensor:
        """Executes symmetric normalized sum propagation without self-loops or non-linearities."""
        row, col = edge_index
        deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(
            edge_index,
            x=x,
            norm=norm,
            is_high=is_high,
        )

        return apply_mask(out, is_high, self.d_low)

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
        is_high_i: torch.Tensor,
        is_high_j: torch.Tensor,
    ) -> torch.Tensor:
        """Routes message and applies symmetric edge normalization factor."""
        msg = self.route_messages(x_j, is_high_i, is_high_j)
        return norm.unsqueeze(-1) * msg


class PANDAGCNConv(BasePANDAConv):
    """PANDA GCN Convolution: Self-loops + Symmetric normalization + Linear + ReLU."""

    def __init__(self, d_low: int, d_high: int) -> None:
        """Initializes GCN transformation weights and sum aggregation."""
        super().__init__(d_low=d_low, d_high=d_high, aggr="add")

        self.lin_low = nn.Linear(d_low, d_low)
        self.lin_high = nn.Linear(d_high, d_high)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_high: torch.Tensor,
    ) -> torch.Tensor:
        """Executes GCN convolution with self-loops, symmetric norm, and ReLU."""
        num_nodes = x.size(0)
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index_with_loops
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(
            edge_index_with_loops,
            x=x,
            norm=norm,
            is_high=is_high,
        )

        transformed = torch.empty_like(out)
        low_mask = ~is_high

        if low_mask.any():
            transformed_low = F.relu(self.lin_low(out[low_mask, : self.d_low]))
            zeros = torch.zeros(
                transformed_low.size(0),
                self.d_high - self.d_low,
                device=x.device,
                dtype=x.dtype,
            )
            transformed[low_mask] = torch.cat([transformed_low, zeros], dim=-1)

        if is_high.any():
            transformed[is_high] = F.relu(self.lin_high(out[is_high]))

        return apply_mask(transformed, is_high, self.d_low)

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
        is_high_i: torch.Tensor,
        is_high_j: torch.Tensor,
    ) -> torch.Tensor:
        """Routes message and normalizes by self-looped degree."""
        msg = self.route_messages(x_j, is_high_i, is_high_j)
        return norm.unsqueeze(-1) * msg


class PANDASAGEConv(BasePANDAConv):
    """PANDA GraphSAGE Convolution: Mean neighbor pooling + Root concatenation."""

    def __init__(self, d_low: int, d_high: int) -> None:
        """Initializes GraphSAGE convolution with mean neighbor aggregation."""
        super().__init__(d_low=d_low, d_high=d_high, aggr="mean")

        self.lin_low = nn.Linear(2 * d_low, d_low)
        self.lin_high = nn.Linear(2 * d_high, d_high)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_high: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates neighbor representations, concatenates with self, and updates."""
        neigh_out = self.propagate(
            edge_index,
            x=x,
            is_high=is_high,
        )

        out = torch.empty_like(x)
        low_mask = ~is_high

        if low_mask.any():
            root_low = x[low_mask, : self.d_low]
            neigh_low = neigh_out[low_mask, : self.d_low]
            cat_low = torch.cat([root_low, neigh_low], dim=-1)
            updated_low = F.relu(self.lin_low(cat_low))
            zeros = torch.zeros(
                updated_low.size(0),
                self.d_high - self.d_low,
                device=x.device,
                dtype=x.dtype,
            )
            out[low_mask] = torch.cat([updated_low, zeros], dim=-1)

        if is_high.any():
            root_high = x[is_high]
            neigh_high = neigh_out[is_high]
            cat_high = torch.cat([root_high, neigh_high], dim=-1)
            out[is_high] = F.relu(self.lin_high(cat_high))

        return apply_mask(out, is_high, self.d_low)

    def message(
        self,
        x_j: torch.Tensor,
        is_high_i: torch.Tensor,
        is_high_j: torch.Tensor,
    ) -> torch.Tensor:
        """Routes message between source and target nodes."""
        return self.route_messages(x_j, is_high_i, is_high_j)


class PANDAGATConv(BasePANDAConv):
    """PANDA GAT Convolution: Multi-head attention scoring with width projection alignment."""

    def __init__(self, d_low: int, d_high: int, heads: int = 2) -> None:
        """Initializes PANDAGATConv with width-aligned multi-head attention."""
        super().__init__(d_low=d_low, d_high=d_high, aggr="add")
        self.heads = heads

        self.att_low = nn.Parameter(torch.empty(heads, 2 * d_low))
        self.att_high = nn.Parameter(torch.empty(heads, 2 * d_high))

        self.leaky_relu = nn.LeakyReLU(0.2)

        self._reset_gat_parameters()

    def _reset_gat_parameters(self) -> None:
        """Initializes GAT attention vectors with Glorot uniform."""
        nn.init.xavier_uniform_(self.att_low)
        nn.init.xavier_uniform_(self.att_high)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_high: torch.Tensor,
    ) -> torch.Tensor:
        """Computes width-aligned attention coefficients and aggregates messages."""
        out = self.propagate(
            edge_index,
            x=x,
            is_high=is_high,
            size=(x.size(0), x.size(0)),
        )

        out = F.elu(out)
        return apply_mask(out, is_high, self.d_low)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
        is_high_i: torch.Tensor,
        is_high_j: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates attention scores and computes weighted routed messages."""
        msg = self.route_messages(x_j, is_high_i, is_high_j)

        num_edges = x_i.size(0)
        low_dst = ~is_high_i
        high_dst = is_high_i

        alpha = torch.zeros(num_edges, self.heads, device=x_i.device, dtype=x_i.dtype)

        if low_dst.any():
            h_i_low = x_i[low_dst, : self.d_low]
            msg_low = msg[low_dst, : self.d_low]
            cat_low = torch.cat([h_i_low, msg_low], dim=-1)
            scores_low = (cat_low.unsqueeze(1) * self.att_low.unsqueeze(0)).sum(dim=-1)
            alpha[low_dst] = self.leaky_relu(scores_low)

        if high_dst.any():
            h_i_high = x_i[high_dst]
            msg_high = msg[high_dst]
            cat_high = torch.cat([h_i_high, msg_high], dim=-1)
            scores_high = (cat_high.unsqueeze(1) * self.att_high.unsqueeze(0)).sum(dim=-1)
            alpha[high_dst] = self.leaky_relu(scores_high)

        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        mean_alpha = alpha.mean(dim=1, keepdim=True)

        return mean_alpha * msg


class PANDAGINConv(BasePANDAConv):
    """PANDA GIN Convolution: Sum aggregation + Learnable epsilon + Width-specific MLPs."""

    def __init__(
        self,
        d_low: int,
        d_high: int,
        eps: float = 0.0,
        train_eps: bool = True,
    ) -> None:
        """Initializes PANDAGINConv with sum aggregation and dual-width MLPs."""
        super().__init__(d_low=d_low, d_high=d_high, aggr="add")
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))

        # Dual 2-layer MLPs for low and high dimensionalities with LayerNorm and ReLU
        self.mlp_low = nn.Sequential(
            nn.Linear(d_low, d_low),
            nn.LayerNorm(d_low),
            nn.ReLU(),
            nn.Linear(d_low, d_low),
        )

        self.mlp_high = nn.Sequential(
            nn.Linear(d_high, d_high),
            nn.LayerNorm(d_high),
            nn.ReLU(),
            nn.Linear(d_high, d_high),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_high: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates neighbor representations with sum, applies (1+eps)*x, and transforms via MLP."""
        neigh_out = self.propagate(
            edge_index,
            x=x,
            is_high=is_high,
        )

        out = torch.empty_like(x)
        low_mask = ~is_high

        if low_mask.any():
            root_low = x[low_mask, : self.d_low]
            neigh_low = neigh_out[low_mask, : self.d_low]
            comb_low = (1.0 + self.eps) * root_low + neigh_low
            updated_low = self.mlp_low(comb_low)
            zeros = torch.zeros(
                updated_low.size(0),
                self.d_high - self.d_low,
                device=x.device,
                dtype=x.dtype,
            )
            out[low_mask] = torch.cat([updated_low, zeros], dim=-1)

        if is_high.any():
            root_high = x[is_high]
            neigh_high = neigh_out[is_high]
            comb_high = (1.0 + self.eps) * root_high + neigh_high
            out[is_high] = self.mlp_high(comb_high)

        return apply_mask(out, is_high, self.d_low)

    def message(
        self,
        x_j: torch.Tensor,
        is_high_i: torch.Tensor,
        is_high_j: torch.Tensor,
    ) -> torch.Tensor:
        """Routes message between source and target nodes."""
        return self.route_messages(x_j, is_high_i, is_high_j)

