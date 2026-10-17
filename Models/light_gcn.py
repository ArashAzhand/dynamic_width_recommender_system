import torch
from torch import nn, Tensor
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class LightGCN(nn.Module):
    def __init__(self, num_users, num_movies, hidden=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # embeddings only, no linear feature projection
        self.user_emb = nn.Embedding(num_users, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden)

        # LightGCN layers (normalized adjacency aggregation without ReLUs/Linear)
        self.convs = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

    def forward(self, edge_index, num_users, num_movies):
        device = self.user_emb.weight.device
        
        user_ids = torch.arange(num_users, device=device)
        movie_ids = torch.arange(num_movies, device=device)

        # initial embeddings E^(0)
        user_x = self.user_emb(user_ids)
        movie_x = self.movie_emb(movie_ids)

        x = torch.cat([user_x, movie_x], dim=0)

        # build bipartite edge index
        user_offset = 0
        movie_offset = num_users

        edge_user = edge_index[0]
        edge_movie = edge_index[1] + movie_offset

        homo_edge_index = torch.stack([
            torch.cat([edge_user, edge_movie]),
            torch.cat([edge_movie, edge_user])
        ])

        # LightGCN propagation
        embs = [x]
        for conv in self.convs:
            x = conv(x, homo_edge_index)
            embs.append(x)

        # average embeddings
        out = torch.stack(embs, dim=0).mean(dim=0)

        user_out = out[:num_users]
        movie_out = out[num_users:]

        return user_out, movie_out


class PANDA_LGConv(MessagePassing):
    def __init__(self, low_dim=64, high_dim=128, normalize=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.normalize = normalize
        self.low_dim = low_dim
        self.high_dim = high_dim
        
        self.W_f = nn.Linear(low_dim, high_dim, bias=False)
        self.W_g = nn.Linear(high_dim, low_dim, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor, is_high_dim: Tensor, edge_weight=None) -> Tensor:
        if self.normalize:
            out = gcn_norm(edge_index, edge_weight, x.size(0), add_self_loops=False, dtype=x.dtype)
            edge_index, edge_weight = out
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, is_high_dim=is_high_dim)

    def message(self, x_j: Tensor, edge_weight: Tensor, is_high_dim_i: Tensor, is_high_dim_j: Tensor) -> Tensor:
        msg = x_j.clone()
        mask_low2high = ((~is_high_dim_j) & is_high_dim_i).squeeze(-1)
        mask_high2low = (is_high_dim_j & (~is_high_dim_i)).squeeze(-1)
        
        if mask_low2high.any():
            low_feats = msg[mask_low2high, :self.low_dim]
            msg[mask_low2high] = self.W_f(low_feats)
            
        if mask_high2low.any():
            high_feats = msg[mask_high2low]
            low_proj = self.W_g(high_feats)
            zeros = torch.zeros(low_proj.size(0), self.high_dim - self.low_dim, device=low_proj.device)
            msg[mask_high2low] = torch.cat([low_proj, zeros], dim=-1)
            
        return msg if edge_weight is None else edge_weight.view(-1, 1) * msg


class PANDAPredictionModule(nn.Module):
    def __init__(self, low_dim=64, high_dim=128):
        super().__init__()
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.W_p = nn.Linear(high_dim, low_dim, bias=False)

    def forward(self, embeddings: Tensor, is_high_dim: Tensor):
        aligned_emb = embeddings[:, :self.low_dim].clone() 
        mask = is_high_dim.squeeze(-1)
        if mask.any():
            high_feats = embeddings[mask]
            aligned_emb[mask] = self.W_p(high_feats)
        return aligned_emb


class PANDA_LightGCN(nn.Module):
    def __init__(self, num_users, num_movies, low_dim=64, high_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.low_dim = low_dim
        self.high_dim = high_dim

        # Pad initial embeddings to high_dim. We only use low_dim unless it's a hub node.
        self.user_emb = nn.Embedding(num_users, high_dim)
        self.movie_emb = nn.Embedding(num_movies, high_dim)

        self.convs = nn.ModuleList([
            PANDA_LGConv(low_dim=low_dim, high_dim=high_dim) for _ in range(num_layers)
        ])
        
        self.predictor = PANDAPredictionModule(low_dim, high_dim)

    def forward(self, edge_index, num_users, num_movies, is_high_dim_all):
        device = self.user_emb.weight.device
        
        user_ids = torch.arange(num_users, device=device)
        movie_ids = torch.arange(num_movies, device=device)

        user_x = self.user_emb(user_ids)
        movie_x = self.movie_emb(movie_ids)
        x = torch.cat([user_x, movie_x], dim=0)

        # Zero out the padded dimensions for low-dim nodes
        low_dim_mask = ~is_high_dim_all.squeeze(-1)
        x[low_dim_mask, self.low_dim:] = 0.0

        user_offset = num_users
        edge_user = edge_index[0]
        edge_movie = edge_index[1] + user_offset

        homo_edge_index = torch.stack([
            torch.cat([edge_user, edge_movie]),
            torch.cat([edge_movie, edge_user])
        ])

        embs = [x]
        for conv in self.convs:
            x = conv(x, homo_edge_index, is_high_dim=is_high_dim_all)
            embs.append(x)

        out = torch.stack(embs, dim=0).mean(dim=0)
        
        # Project 128-dim embeddings back down to 64-dim
        aligned_out = self.predictor(out, is_high_dim_all)

        user_out = aligned_out[:num_users]
        movie_out = aligned_out[num_users:]

        return user_out, movie_out
