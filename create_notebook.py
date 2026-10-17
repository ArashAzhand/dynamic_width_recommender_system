import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Recommender System Data Verification\n",
    "This notebook verifies the graph properties, splits, and distributions of the MovieLens 100k dataset used in our pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Local imports\n",
    "from Utils.data_utils import load_ml_100k_data, preprocess_users, preprocess_items, load_ml_100k_fold, create_heterodata\n",
    "\n",
    "DATA_DIR = \"Data/ml-100k/\"\n",
    "users_df, items_df, _ = load_ml_100k_data(DATA_DIR)\n",
    "proc_users_df = preprocess_users(users_df)\n",
    "proc_items_df = preprocess_items(items_df)\n",
    "\n",
    "train_df, test_df = load_ml_100k_fold(DATA_DIR, fold_idx=1)\n",
    "\n",
    "hetero_train = create_heterodata(proc_users_df, proc_items_df, train_df)\n",
    "hetero_test  = create_heterodata(proc_users_df, proc_items_df, test_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Contiguous Integer Mapping\n",
    "Verify that User IDs and Movie IDs are mapped from `0` to `N-1`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "user_idx_min, user_idx_max = hetero_train['user', 'rates', 'movie'].edge_index[0].min(), hetero_train['user', 'rates', 'movie'].edge_index[0].max()\n",
    "movie_idx_min, movie_idx_max = hetero_train['user', 'rates', 'movie'].edge_index[1].min(), hetero_train['user', 'rates', 'movie'].edge_index[1].max()\n",
    "\n",
    "print(f\"Train User indices min: {user_idx_min}, max: {user_idx_max}, num nodes: {hetero_train['user'].num_nodes}\")\n",
    "print(f\"Train Movie indices min: {movie_idx_min}, max: {movie_idx_max}, num nodes: {hetero_train['movie'].num_nodes}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Edge Index Shape & Splits\n",
    "Verify that `edge_index` has shape `[2, num_edges]` and check the data split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Train edge_index shape: {hetero_train['user', 'rates', 'movie'].edge_index.shape}\")\n",
    "print(f\"Test edge_index shape: {hetero_test['user', 'rates', 'movie'].edge_index.shape}\")\n",
    "\n",
    "print(\"\\n--- Validation Split Check ---\")\n",
    "print(\"Your code correctly uses the predefined 5-fold cross-validation splits from ML-100k, resulting in an ~80/20 train/test split. Note that an explicit validation set (e.g., 80/10/10 split) is not currently implemented in `train.py`.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Data Leakage\n",
    "Verify that no test edges exist in the training graph."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert edge index to set of tuples for fast intersection\n",
    "train_edges = set(zip(hetero_train['user', 'rates', 'movie'].edge_index[0].numpy(), hetero_train['user', 'rates', 'movie'].edge_index[1].numpy()))\n",
    "test_edges = set(zip(hetero_test['user', 'rates', 'movie'].edge_index[0].numpy(), hetero_test['user', 'rates', 'movie'].edge_index[1].numpy()))\n",
    "\n",
    "intersection = train_edges.intersection(test_edges)\n",
    "print(f\"Number of edges in both train and test: {len(intersection)}\")\n",
    "if len(intersection) == 0:\n",
    "    print(\"Verification passed: No data leakage.\")\n",
    "else:\n",
    "    print(\"Verification failed: Data leakage detected!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Undirected Graph\n",
    "Verify that for every `u -> i` edge, there is an `i -> u` edge."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_edges_fwd = set(zip(hetero_train['user', 'rates', 'movie'].edge_index[0].numpy(), hetero_train['user', 'rates', 'movie'].edge_index[1].numpy()))\n",
    "train_edges_rev = set(zip(hetero_train['movie', 'rated_by', 'user'].edge_index[1].numpy(), hetero_train['movie', 'rated_by', 'user'].edge_index[0].numpy()))\n",
    "\n",
    "if train_edges_fwd == train_edges_rev:\n",
    "    print(\"Verification passed: Graph is undirected (reverse edges match forward edges).\")\n",
    "else:\n",
    "    print(\"Verification failed: Graph is not undirected properly.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Graph Statistics & Degree Distribution\n",
    "Plot and observe the highly skewed degree distribution of users and movies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "user_degrees = torch.bincount(hetero_train['user', 'rates', 'movie'].edge_index[0])\n",
    "movie_degrees = torch.bincount(hetero_train['user', 'rates', 'movie'].edge_index[1])\n",
    "\n",
    "print(f\"User degree - mean: {user_degrees.float().mean():.2f}, min: {user_degrees.min()}, max: {user_degrees.max()}\")\n",
    "print(f\"Movie degree - mean: {movie_degrees.float().mean():.2f}, min: {movie_degrees.min()}, max: {movie_degrees.max()}\")\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "sns.histplot(user_degrees.numpy(), bins=50, ax=axes[0], color='blue', kde=True)\n",
    "axes[0].set_title(\"User Degree Distribution\")\n",
    "axes[0].set_xlabel(\"Degree\")\n",
    "axes[0].set_ylabel(\"Count\")\n",
    "\n",
    "sns.histplot(movie_degrees.numpy(), bins=50, ax=axes[1], color='orange', kde=True)\n",
    "axes[1].set_title(\"Movie Degree Distribution\")\n",
    "axes[1].set_xlabel(\"Degree\")\n",
    "axes[1].set_ylabel(\"Count\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Verification.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
