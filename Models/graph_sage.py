import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_user, in_movie, num_users, num_movies, hidden=64):
        super().__init__()

        self.user_lin = nn.Linear(in_user, hidden)
        self.movie_lin = nn.Linear(in_movie, hidden)
        self.dropout = nn.Dropout(0)
        self.user_emb = nn.Embedding(num_users, hidden)
        self.movie_emb = nn.Embedding(num_movies, hidden) 


        # Layer 1
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((hidden, hidden), hidden),
            ('movie', 'rated_by', 'user'): SAGEConv((hidden, hidden), hidden),
        })

        # Layer 2
        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((hidden, hidden), hidden),
            ('movie', 'rated_by', 'user'): SAGEConv((hidden, hidden), hidden),
        })


    def forward(self, data):
        device = data['user'].x.device

        num_users = data['user'].num_nodes
        num_movies = data['movie'].num_nodes

        user_ids = torch.arange(num_users, device=device)
        movie_ids = torch.arange(num_movies, device=device)

        user_x = self.user_lin(data['user'].x) + self.user_emb(user_ids)
        movie_x = self.movie_lin(data['movie'].x) + self.movie_emb(movie_ids)

        x_dict = {
            'user': user_x,
            'movie': movie_x
        }

        # ---- Layer 1 ----
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            key: self.dropout(torch.relu(x))
            for key, x in x_dict.items()
        }

        # ---- Layer 2 ----
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {
            key: self.dropout(torch.relu(x))
            for key, x in x_dict.items()
        }

        user_x = x_dict['user']
        movie_x = x_dict['movie']

        return user_x, movie_x
