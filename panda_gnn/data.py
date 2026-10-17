"""Data Pipeline and Hygiene Module for Bipartite Collaborative Filtering."""

from dataclasses import dataclass
import os
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable container for bipartite graph splits and indexing metadata."""

    train_edge_index: torch.Tensor
    train_user_item_edges: torch.Tensor
    val_user_item_edges: torch.Tensor
    test_user_item_edges: torch.Tensor
    num_users: int
    num_items: int
    num_total_nodes: int
    user_train_dict: Dict[int, Set[int]]
    user_val_dict: Dict[int, Set[int]]
    user_test_dict: Dict[int, Set[int]]


class MovieLens100KLoader:
    """Data ingestion and preprocessing pipeline for MovieLens-100K."""

    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(
        self,
        data_dir: str = "Data/ml-100k",
        implicit_threshold: int = 3,
        seed: int = 42,
    ) -> None:
        """Initializes data loader parameters."""
        self.data_dir = data_dir
        self.implicit_threshold = implicit_threshold
        self.seed = seed

    def _ensure_downloaded(self) -> str:
        """Ensures MovieLens 100K dataset is downloaded and extracted."""
        data_file = os.path.join(self.data_dir, "u.data")
        if os.path.exists(data_file):
            return data_file

        import urllib.request
        import zipfile

        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "ml-100k.zip")
        print(f"Downloading MovieLens-100K dataset from {self.DATA_URL}...")
        urllib.request.urlretrieve(self.DATA_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.data_dir))

        if not os.path.exists(data_file):
            extracted_nested = os.path.join(os.path.dirname(self.data_dir), "ml-100k", "u.data")
            if os.path.exists(extracted_nested):
                data_file = extracted_nested

        return data_file

    def load_and_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Loads dataset, converts to implicit positive edges, re-indexes, and splits."""
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Splits must sum to 1.0")

        file_path = self._ensure_downloaded()
        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )

        # 1. Implicit feedback thresholding
        positive_df = df[df["rating"] >= self.implicit_threshold].copy()

        # 2. Contiguous 0-indexed ID mapping
        unique_users = sorted(positive_df["user_id"].unique())
        unique_items = sorted(positive_df["item_id"].unique())

        user2id = {orig: idx for idx, orig in enumerate(unique_users)}
        item2id = {orig: idx for idx, orig in enumerate(unique_items)}

        positive_df["u_idx"] = positive_df["user_id"].map(user2id)
        positive_df["i_idx"] = positive_df["item_id"].map(item2id)

        num_users = len(user2id)
        num_items = len(item2id)
        num_total_nodes = num_users + num_items

        rng = np.random.RandomState(self.seed)

        train_records = []
        val_records = []
        test_records = []

        user_train_dict: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        user_val_dict: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        user_test_dict: Dict[int, Set[int]] = {u: set() for u in range(num_users)}

        for u_idx, group in positive_df.groupby("u_idx"):
            items = group["i_idx"].to_numpy()
            n_items = len(items)

            if n_items == 1:
                train_records.append((u_idx, items[0]))
                user_train_dict[u_idx].add(items[0])
            elif n_items == 2:
                train_records.append((u_idx, items[0]))
                test_records.append((u_idx, items[1]))
                user_train_dict[u_idx].add(items[0])
                user_test_dict[u_idx].add(items[1])
            else:
                shuffled_indices = rng.permutation(n_items)
                items_shuffled = items[shuffled_indices]

                n_test = max(1, int(round(n_items * test_ratio)))
                n_val = max(1, int(round(n_items * val_ratio)))
                n_train = n_items - n_val - n_test

                if n_train < 1:
                    n_train = 1
                    if n_val > 1:
                        n_val -= 1
                    else:
                        n_test -= 1

                train_items = items_shuffled[:n_train]
                val_items = items_shuffled[n_train : n_train + n_val]
                test_items = items_shuffled[n_train + n_val :]

                for item in train_items:
                    train_records.append((u_idx, item))
                    user_train_dict[u_idx].add(item)
                for item in val_items:
                    val_records.append((u_idx, item))
                    user_val_dict[u_idx].add(item)
                for item in test_items:
                    test_records.append((u_idx, item))
                    user_test_dict[u_idx].add(item)

        train_u = torch.tensor([r[0] for r in train_records], dtype=torch.long)
        train_i = torch.tensor([r[1] for r in train_records], dtype=torch.long)
        train_user_item_edges = torch.stack([train_u, train_i], dim=0)

        val_u = torch.tensor([r[0] for r in val_records], dtype=torch.long)
        val_i = torch.tensor([r[1] for r in val_records], dtype=torch.long)
        val_user_item_edges = torch.stack([val_u, val_i], dim=0)

        test_u = torch.tensor([r[0] for r in test_records], dtype=torch.long)
        test_i = torch.tensor([r[1] for r in test_records], dtype=torch.long)
        test_user_item_edges = torch.stack([test_u, test_i], dim=0)

        shifted_items = train_i + num_users
        u_to_i = torch.stack([train_u, shifted_items], dim=0)
        i_to_u = torch.stack([shifted_items, train_u], dim=0)
        train_edge_index = torch.cat([u_to_i, i_to_u], dim=1)

        return DatasetSplits(
            train_edge_index=train_edge_index,
            train_user_item_edges=train_user_item_edges,
            val_user_item_edges=val_user_item_edges,
            test_user_item_edges=test_user_item_edges,
            num_users=num_users,
            num_items=num_items,
            num_total_nodes=num_total_nodes,
            user_train_dict=user_train_dict,
            user_val_dict=user_val_dict,
            user_test_dict=user_test_dict,
        )
