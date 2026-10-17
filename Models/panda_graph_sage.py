import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv
from Models.light_gcn import PANDAPredictionModule

from torch_geometric.nn.conv import MessagePassing

class PANDA_SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, low_dim=64, high_dim=128, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        
        self.W_f = nn.Linear(low_dim, high_dim, bias=False)
        self.W_g = nn.Linear(high_dim, low_dim, bias=False)

    def forward(self, x, edge_index, is_high_dim):
        if isinstance(is_high_dim, tuple):
            is_high_dim_src, is_high_dim_dst = is_high_dim
        else:
            is_high_dim_src = is_high_dim_dst = is_high_dim
            
        out = self.propagate(edge_index, x=x, is_high_dim=(is_high_dim_src, is_high_dim_dst))
        
        if isinstance(x, tuple):
            x_dst = x[1]
        else:
            x_dst = x
            
        out = self.lin_l(x_dst) + self.lin_r(out)
        
        # Enforce zeros on the extra dimensions for low-dim nodes
        if (~is_high_dim_dst).any():
            out = torch.cat([out[:, :self.low_dim], out[:, self.low_dim:] * is_high_dim_dst.float()], dim=-1)
            
        return out

    def message(self, x_j, is_high_dim_i, is_high_dim_j):
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
            
        return msg


class PANDA_GraphSAGE(nn.Module):
    def __init__(self, in_user, in_movie, num_users, num_movies, low_dim=64, high_dim=128):
        super().__init__()
        self.low_dim = low_dim
        self.high_dim = high_dim

        # Project node features directly to high_dim, we will mask later
        self.user_lin = nn.Linear(in_user, high_dim)
        self.movie_lin = nn.Linear(in_movie, high_dim)
        self.dropout = nn.Dropout(0)
        
        # Initial embeddings also in high_dim
        self.user_emb = nn.Embedding(num_users, high_dim)
        self.movie_emb = nn.Embedding(num_movies, high_dim) 

        # Layer 1
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): PANDA_SAGEConv(high_dim, high_dim, low_dim=low_dim, high_dim=high_dim),
            ('movie', 'rated_by', 'user'): PANDA_SAGEConv(high_dim, high_dim, low_dim=low_dim, high_dim=high_dim),
        })

        # Layer 2
        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): PANDA_SAGEConv(high_dim, high_dim, low_dim=low_dim, high_dim=high_dim),
            ('movie', 'rated_by', 'user'): PANDA_SAGEConv(high_dim, high_dim, low_dim=low_dim, high_dim=high_dim),
        })
        
        self.pred_module = PANDAPredictionModule(low_dim=low_dim, high_dim=high_dim)

    def forward(self, data, is_high_dim_dict):
        device = data['user'].x.device

        num_users = data['user'].num_nodes
        num_movies = data['movie'].num_nodes

        user_ids = torch.arange(num_users, device=device)
        movie_ids = torch.arange(num_movies, device=device)

        user_x = self.user_lin(data['user'].x) + self.user_emb(user_ids)
        movie_x = self.movie_lin(data['movie'].x) + self.movie_emb(movie_ids)
        
        def apply_mask(x, mask, low_dim):
            if (~mask).any():
                return torch.cat([x[:, :low_dim], x[:, low_dim:] * mask.float()], dim=-1)
            return x

        # Enforce initial mask!
        user_x = apply_mask(user_x, is_high_dim_dict['user'], self.low_dim)
        movie_x = apply_mask(movie_x, is_high_dim_dict['movie'], self.low_dim)

        x_dict = {
            'user': user_x,
            'movie': movie_x
        }

        # ---- Layer 1 ----
        x_dict = self.conv1(x_dict, data.edge_index_dict, is_high_dim_dict=is_high_dim_dict)
        x_dict = {
            key: self.dropout(torch.relu(x))
            for key, x in x_dict.items()
        }
        
        # Enforce mask after relu
        x_dict['user'] = apply_mask(x_dict['user'], is_high_dim_dict['user'], self.low_dim)
        x_dict['movie'] = apply_mask(x_dict['movie'], is_high_dim_dict['movie'], self.low_dim)

        # ---- Layer 2 ----
        x_dict = self.conv2(x_dict, data.edge_index_dict, is_high_dim_dict=is_high_dim_dict)
        x_dict = {
            key: self.dropout(torch.relu(x))
            for key, x in x_dict.items()
        }
        
        x_dict['user'] = apply_mask(x_dict['user'], is_high_dim_dict['user'], self.low_dim)
        x_dict['movie'] = apply_mask(x_dict['movie'], is_high_dim_dict['movie'], self.low_dim)

        # Use PANDA Prediction Module
        user_out = self.pred_module(x_dict['user'], is_high_dim_dict['user'])
        movie_out = self.pred_module(x_dict['movie'], is_high_dim_dict['movie'])

        return user_out, movie_out
