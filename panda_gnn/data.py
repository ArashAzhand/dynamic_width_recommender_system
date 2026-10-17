"""Data Pipeline and Hygiene Module for Bipartite Collaborative Filtering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import os
import ssl
from typing import Dict, Optional, Set, Tuple
import urllib.request
import zipfile

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


class DatasetType(str, Enum):
    """Supported benchmark recommendation datasets."""

    ML_100K = "ML-100K"
    ML_1M = "ML-1M"


class BaseDatasetLoader(ABC):
    """Abstract base class for recommendation dataset ingestion and leak-proof splitting."""

    def __init__(
        self,
        data_dir: str,
        implicit_threshold: int = 3,
        seed: int = 42,
    ) -> None:
        """Initializes loader parameters."""
        self.data_dir = data_dir
        self.implicit_threshold = implicit_threshold
        self.seed = seed

    @abstractmethod
    def load_and_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Abstract method loading data and returning partitioned splits."""
        pass

    @abstractmethod
    def load_kfold_splits(
        self,
        num_folds: int = 5,
        fold_idx: int = 0,
        val_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Abstract method loading data and returning K-fold partitioned splits for fold_idx."""
        pass

    def _create_splits_from_df(
        self,
        positive_df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Contiguously maps IDs, creates leak-proof per-user 80/10/10 splits, and constructs shifted edges."""
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Splits must sum to 1.0")

        # 1. Contiguous 0-indexed ID mapping
        unique_users = sorted(positive_df["user_id"].unique())
        unique_items = sorted(positive_df["item_id"].unique())

        user2id = {orig: idx for idx, orig in enumerate(unique_users)}
        item2id = {orig: idx for idx, orig in enumerate(unique_items)}

        positive_df = positive_df.copy()
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

        # Shift item IDs by num_users for homogeneous message passing
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

    def _create_kfold_splits_from_df(
        self,
        positive_df: pd.DataFrame,
        num_folds: int = 5,
        fold_idx: int = 0,
        val_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Creates a rigorous K-fold cross-validation split for fold_idx."""
        if fold_idx < 0 or fold_idx >= num_folds:
            raise ValueError(f"fold_idx must be in [0, {num_folds - 1}], got {fold_idx}")

        unique_users = sorted(positive_df["user_id"].unique())
        unique_items = sorted(positive_df["item_id"].unique())

        user2id = {orig: idx for idx, orig in enumerate(unique_users)}
        item2id = {orig: idx for idx, orig in enumerate(unique_items)}

        positive_df = positive_df.copy()
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

            if n_items < num_folds:
                # Sparse interaction handling: ensure training graph connectivity
                if n_items == 1:
                    train_records.append((u_idx, items[0]))
                    user_train_dict[u_idx].add(items[0])
                else:
                    perm = rng.permutation(items)
                    test_item = perm[fold_idx % n_items]
                    test_records.append((u_idx, test_item))
                    user_test_dict[u_idx].add(test_item)
                    for itm in perm:
                        if itm != test_item:
                            train_records.append((u_idx, itm))
                            user_train_dict[u_idx].add(itm)
            else:
                shuffled_items = rng.permutation(items)
                folds = np.array_split(shuffled_items, num_folds)
                test_items = folds[fold_idx]

                remaining = [itm for f_i, f_items in enumerate(folds) if f_i != fold_idx for itm in f_items]
                n_val = max(1, int(round(len(remaining) * val_ratio)))
                val_items = remaining[-n_val:]
                train_items = remaining[:-n_val]
                if len(train_items) == 0:
                    train_items = val_items
                    val_items = []

                for itm in test_items:
                    test_records.append((u_idx, itm))
                    user_test_dict[u_idx].add(itm)
                for itm in val_items:
                    val_records.append((u_idx, itm))
                    user_val_dict[u_idx].add(itm)
                for itm in train_items:
                    train_records.append((u_idx, itm))
                    user_train_dict[u_idx].add(itm)

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


class MovieLens100KLoader(BaseDatasetLoader):
    """Data ingestion and preprocessing pipeline for MovieLens-100K."""

    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    def __init__(
        self,
        data_dir: str = "Data/ml-100k",
        implicit_threshold: int = 3,
        seed: int = 42,
    ) -> None:
        """Initializes loader."""
        super().__init__(data_dir=data_dir, implicit_threshold=implicit_threshold, seed=seed)

    def _ensure_downloaded(self) -> str:
        """Ensures MovieLens 100K dataset is downloaded and extracted."""
        data_file = os.path.join(self.data_dir, "u.data")
        if os.path.exists(data_file):
            return data_file

        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "ml-100k.zip")
        print(f"Downloading MovieLens-100K dataset from {self.DATA_URL}...")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(self.DATA_URL, context=ctx) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.data_dir))

        if not os.path.exists(data_file):
            nested = os.path.join(os.path.dirname(self.data_dir), "ml-100k", "u.data")
            if os.path.exists(nested):
                data_file = nested

        return data_file

    def load_and_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Loads MovieLens-100K, filters by threshold, and generates splits."""
        file_path = self._ensure_downloaded()
        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        positive_df = df[df["rating"] >= self.implicit_threshold].copy()
        return self._create_splits_from_df(positive_df, train_ratio, val_ratio, test_ratio)

    def load_kfold_splits(
        self,
        num_folds: int = 5,
        fold_idx: int = 0,
        val_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Loads MovieLens-100K and generates K-fold partitioned splits for fold_idx."""
        file_path = self._ensure_downloaded()
        df = pd.read_csv(
            file_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        positive_df = df[df["rating"] >= self.implicit_threshold].copy()
        return self._create_kfold_splits_from_df(positive_df, num_folds, fold_idx, val_ratio)


class MovieLens1MLoader(BaseDatasetLoader):
    """Data ingestion and preprocessing pipeline for MovieLens-1M."""

    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(
        self,
        data_dir: str = "Data/ml-1m",
        implicit_threshold: int = 3,
        seed: int = 42,
    ) -> None:
        """Initializes loader."""
        super().__init__(data_dir=data_dir, implicit_threshold=implicit_threshold, seed=seed)

    def _ensure_downloaded(self) -> str:
        """Ensures MovieLens 1M dataset is downloaded and extracted."""
        data_file = os.path.join(self.data_dir, "ratings.dat")
        if os.path.exists(data_file):
            return data_file

        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "ml-1m.zip")
        print(f"Downloading MovieLens-1M dataset from {self.DATA_URL}...")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(self.DATA_URL, context=ctx) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.data_dir))

        if not os.path.exists(data_file):
            nested = os.path.join(os.path.dirname(self.data_dir), "ml-1m", "ratings.dat")
            if os.path.exists(nested):
                data_file = nested

        return data_file

    def load_and_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Loads MovieLens-1M, filters by threshold, and generates splits."""
        file_path = self._ensure_downloaded()
        df = pd.read_csv(
            file_path,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        positive_df = df[df["rating"] >= self.implicit_threshold].copy()
        return self._create_splits_from_df(positive_df, train_ratio, val_ratio, test_ratio)

    def load_kfold_splits(
        self,
        num_folds: int = 5,
        fold_idx: int = 0,
        val_ratio: float = 0.1,
    ) -> DatasetSplits:
        """Loads MovieLens-1M and generates K-fold partitioned splits for fold_idx."""
        file_path = self._ensure_downloaded()
        df = pd.read_csv(
            file_path,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        positive_df = df[df["rating"] >= self.implicit_threshold].copy()
        return self._create_kfold_splits_from_df(positive_df, num_folds, fold_idx, val_ratio)


class DatasetFactory:
    """Factory creating appropriate dataset loader instance."""

    @staticmethod
    def get_loader(
        dataset_name: str,
        data_dir: Optional[str] = None,
        implicit_threshold: int = 3,
        seed: int = 42,
    ) -> BaseDatasetLoader:
        """Instantiates dataset loader by name."""
        name_clean = dataset_name.strip().upper().replace("_", "-")
        if name_clean in ("ML-100K", "MOVIELENS-100K", "100K"):
            target_dir = data_dir or "Data/ml-100k"
            return MovieLens100KLoader(data_dir=target_dir, implicit_threshold=implicit_threshold, seed=seed)
        elif name_clean in ("ML-1M", "MOVIELENS-1M", "1M"):
            target_dir = data_dir or "Data/ml-1m"
            return MovieLens1MLoader(data_dir=target_dir, implicit_threshold=implicit_threshold, seed=seed)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: ML-100K, ML-1M")
