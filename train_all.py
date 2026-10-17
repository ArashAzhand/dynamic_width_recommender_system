import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np

# Local imports
from Utils.data_utils import load_ml_100k_data, preprocess_users, preprocess_items, load_ml_100k_fold, create_heterodata
from Models.light_gcn import LightGCN, PANDA_LightGCN
from Models.graph_sage import GraphSAGE
from Models.graph_attention import GAT
from Models.panda_graph_sage import PANDA_GraphSAGE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Data/ml-100k/"
EPOCHS = 200
LR = 1e-3
K_PERCENTILE = 0.10

print("\n--- Loading and Preprocessing Data ---")
users_df, items_df, _ = load_ml_100k_data(DATA_DIR)
proc_users_df = preprocess_users(users_df)
proc_items_df = preprocess_items(items_df)

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    return loss

def sample_negative_items(edge_index, num_movies):
    num_edges = edge_index.size(1)
    neg_items = torch.randint(0, num_movies, (num_edges,), device=edge_index.device)
    return neg_items

def evaluate_metrics_at_k(model, train_data, test_edge_index, num_users, num_movies, is_panda=False, is_high_dim_all=None, is_hetero=False, is_panda_hetero=False, is_high_dim_dict=None, k=20):
    model.eval()
    train_edge_index = train_data['user', 'rates', 'movie'].edge_index
    with torch.no_grad():
        if is_panda_hetero:
            user_emb, movie_emb = model(train_data, is_high_dim_dict)
        elif is_panda:
            user_emb, movie_emb = model(train_edge_index, num_users, num_movies, is_high_dim_all)
        elif is_hetero:
            user_emb, movie_emb = model(train_data)
        else:
            user_emb, movie_emb = model(train_edge_index, num_users, num_movies)
            
        all_scores = torch.matmul(user_emb, movie_emb.t())
        all_scores[train_edge_index[0], train_edge_index[1]] = -float('inf')
        
        _, top_k_items = torch.topk(all_scores, k=k, dim=1)
        top_k_items = top_k_items.cpu().numpy()
        
        test_u = test_edge_index[0].cpu().numpy()
        test_i = test_edge_index[1].cpu().numpy()
        
        from collections import defaultdict
        test_user_items = defaultdict(set)
        for u, i in zip(test_u, test_i):
            test_user_items[u].add(i)
            
        recalls = []
        ndcgs = []
        for u, true_items in test_user_items.items():
            pred_items = top_k_items[u]
            hits = 0
            dcg = 0.0
            for idx, item in enumerate(pred_items):
                if item in true_items:
                    hits += 1
                    dcg += 1.0 / np.log2(idx + 2.0)
            
            idcg = sum(1.0 / np.log2(idx + 2.0) for idx in range(min(k, len(true_items))))
            
            recalls.append(hits / len(true_items))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            
        return np.mean(recalls), np.mean(ndcgs)

def get_panda_masks(edge_index, num_users, num_movies, k_percentile=0.10):
    user_degrees = torch.bincount(edge_index[0], minlength=num_users).float()
    item_degrees = torch.bincount(edge_index[1], minlength=num_movies).float()
    
    q = 1.0 - k_percentile
    user_cutoff = torch.quantile(user_degrees, q)
    item_cutoff = torch.quantile(item_degrees, q)
    
    is_high_dim_user = user_degrees >= user_cutoff
    is_high_dim_item = item_degrees >= item_cutoff
    
    return is_high_dim_user, is_high_dim_item

fold = 1
print(f"\n============================")
print(f"Comparison Run Fold {fold} (All Models)")
print(f"============================")

train_df, test_df = load_ml_100k_fold(DATA_DIR, fold_idx=fold)

hetero_train = create_heterodata(proc_users_df, proc_items_df, train_df)
hetero_test  = create_heterodata(proc_users_df, proc_items_df, test_df)

train_loader = DataLoader([hetero_train], batch_size=1)
test_loader = DataLoader([hetero_test], batch_size=1)

train_data = next(iter(train_loader)).to(DEVICE)
test_data = next(iter(test_loader)).to(DEVICE)
test_edge_index = test_data['user', 'rates', 'movie'].edge_index
train_edge_index_full = train_data['user', 'rates', 'movie'].edge_index

num_users = train_data['user'].num_nodes
num_movies = train_data['movie'].num_nodes
in_user = train_data['user'].x.shape[1]
in_movie = train_data['movie'].x.shape[1]

is_high_dim_user, is_high_dim_item = get_panda_masks(train_edge_index_full, num_users, num_movies, k_percentile=K_PERCENTILE)
is_high_dim_all = torch.cat([is_high_dim_user, is_high_dim_item]).view(-1, 1).to(DEVICE)
is_high_dim_dict = {
    'user': is_high_dim_user.view(-1, 1).to(DEVICE),
    'movie': is_high_dim_item.view(-1, 1).to(DEVICE)
}

baseline_model_64 = LightGCN(num_users=num_users, num_movies=num_movies, hidden=64, num_layers=3).to(DEVICE)
baseline_model_128 = LightGCN(num_users=num_users, num_movies=num_movies, hidden=128, num_layers=3).to(DEVICE)
panda_model = PANDA_LightGCN(num_users=num_users, num_movies=num_movies, low_dim=64, high_dim=128, num_layers=3).to(DEVICE)
sage_model = GraphSAGE(in_user, in_movie, num_users, num_movies, hidden=64).to(DEVICE)
gat_model = GAT(in_user, in_movie, num_users, num_movies, hidden=64, heads=2).to(DEVICE)
panda_sage_model = PANDA_GraphSAGE(in_user, in_movie, num_users, num_movies, low_dim=64, high_dim=128).to(DEVICE)

optimizer_base_64 = optim.Adam(baseline_model_64.parameters(), lr=LR)
optimizer_base_128 = optim.Adam(baseline_model_128.parameters(), lr=LR)
optimizer_panda = optim.Adam(panda_model.parameters(), lr=LR)
optimizer_sage = optim.Adam(sage_model.parameters(), lr=LR)
optimizer_gat = optim.Adam(gat_model.parameters(), lr=LR)
optimizer_panda_sage = optim.Adam(panda_sage_model.parameters(), lr=LR)

print("\n--- Training Started ---")

for epoch in range(1, EPOCHS + 1):
    users = train_edge_index_full[0]
    pos_items = train_edge_index_full[1]
    neg_items = sample_negative_items(train_edge_index_full, num_movies)
    
    # Train Baseline 64
    baseline_model_64.train()
    optimizer_base_64.zero_grad()
    u_emb, m_emb = baseline_model_64(train_edge_index_full, num_users, num_movies)
    loss_base_64 = bpr_loss(u_emb[users], m_emb[pos_items], m_emb[neg_items])
    loss_base_64.backward()
    optimizer_base_64.step()
    
    # Train Baseline 128
    baseline_model_128.train()
    optimizer_base_128.zero_grad()
    u_emb, m_emb = baseline_model_128(train_edge_index_full, num_users, num_movies)
    loss_base_128 = bpr_loss(u_emb[users], m_emb[pos_items], m_emb[neg_items])
    loss_base_128.backward()
    optimizer_base_128.step()
    
    # Train PANDA
    panda_model.train()
    optimizer_panda.zero_grad()
    u_emb, m_emb = panda_model(train_edge_index_full, num_users, num_movies, is_high_dim_all)
    loss_panda = bpr_loss(u_emb[users], m_emb[pos_items], m_emb[neg_items])
    loss_panda.backward()
    optimizer_panda.step()

    # Train GraphSAGE
    sage_model.train()
    optimizer_sage.zero_grad()
    u_emb, m_emb = sage_model(train_data)
    loss_sage = bpr_loss(u_emb[users], m_emb[pos_items], m_emb[neg_items])
    loss_sage.backward()
    optimizer_sage.step()

    # Train GAT
    gat_model.train()
    optimizer_gat.zero_grad()
    u_emb, m_emb = gat_model(train_data)
    loss_gat = bpr_loss(u_emb[users], m_emb[pos_items], m_emb[neg_items])
    loss_gat.backward()
    optimizer_gat.step()
    
    # Train PANDA GraphSAGE
    panda_sage_model.train()
    optimizer_panda_sage.zero_grad()
    u_emb, m_emb = panda_sage_model(train_data, is_high_dim_dict)
    loss_panda_sage = bpr_loss(u_emb[users], m_emb[pos_items], m_emb[neg_items])
    loss_panda_sage.backward()
    optimizer_panda_sage.step()
    
    if epoch % 25 == 0:
        rec_64, ndcg_64 = evaluate_metrics_at_k(baseline_model_64, train_data, test_edge_index, num_users, num_movies, k=20)
        rec_128, ndcg_128 = evaluate_metrics_at_k(baseline_model_128, train_data, test_edge_index, num_users, num_movies, k=20)
        rec_panda, ndcg_panda = evaluate_metrics_at_k(panda_model, train_data, test_edge_index, num_users, num_movies, is_panda=True, is_high_dim_all=is_high_dim_all, k=20)
        rec_sage, ndcg_sage = evaluate_metrics_at_k(sage_model, train_data, test_edge_index, num_users, num_movies, is_hetero=True, k=20)
        rec_gat, ndcg_gat = evaluate_metrics_at_k(gat_model, train_data, test_edge_index, num_users, num_movies, is_hetero=True, k=20)
        rec_p_sage, ndcg_p_sage = evaluate_metrics_at_k(panda_sage_model, train_data, test_edge_index, num_users, num_movies, is_panda_hetero=True, is_high_dim_dict=is_high_dim_dict, k=20)
        
        print(f"Epoch {epoch:03d} | Base 64   -> Loss: {loss_base_64.item():.4f}, R@20: {rec_64:.4f}, NDCG@20: {ndcg_64:.4f}")
        print(f"          | Base 128  -> Loss: {loss_base_128.item():.4f}, R@20: {rec_128:.4f}, NDCG@20: {ndcg_128:.4f}")
        print(f"          | PANDA     -> Loss: {loss_panda.item():.4f}, R@20: {rec_panda:.4f}, NDCG@20: {ndcg_panda:.4f}")
        print(f"          | GraphSAGE -> Loss: {loss_sage.item():.4f}, R@20: {rec_sage:.4f}, NDCG@20: {ndcg_sage:.4f}")
        print(f"          | GAT       -> Loss: {loss_gat.item():.4f}, R@20: {rec_gat:.4f}, NDCG@20: {ndcg_gat:.4f}")
        print(f"          | P-SAGE    -> Loss: {loss_panda_sage.item():.4f}, R@20: {rec_p_sage:.4f}, NDCG@20: {ndcg_p_sage:.4f}")
        print("-" * 80)

print(f"\nTraining completed.")
