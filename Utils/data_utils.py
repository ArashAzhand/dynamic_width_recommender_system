import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import os
from sklearn.preprocessing import StandardScaler



def load_ml_100k_data(data_dir="Data/ml-100k/"):
    print("Loading MovieLens 100k data...")

    # file paths
    user_data_path = os.path.join(data_dir, "u.user")
    item_data_path = os.path.join(data_dir, "u.item")
    rating_data_path = os.path.join(data_dir, "u.data")

    # Load user data
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users_df = pd.read_csv(user_data_path, sep='|', names=user_cols, encoding='latin-1')
    print(f"Loaded {len(users_df)} users.")

    # Load movie data
    item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + \
                [f'genre_{i}' for i in range(19)] # There are 19 genre columns
    items_df = pd.read_csv(item_data_path, sep='|', names=item_cols, encoding='ISO-8859-1', engine='python')
    print(f"Loaded {len(items_df)} items (movies).")

    # Load rating data
    rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(rating_data_path, sep='\t', names=rating_cols, encoding='latin-1')
    print(f"Loaded {len(ratings_df)} ratings.")

    return users_df, items_df, ratings_df


def load_ml_100k_fold(data_dir="Data/ml-100k/", fold_idx=1):
    assert 1 <= fold_idx <= 5, "fold_idx must be between 1 and 5"

    train_path = os.path.join(data_dir, f"u{fold_idx}.base")
    test_path  = os.path.join(data_dir, f"u{fold_idx}.test")

    rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

    train_df = pd.read_csv(train_path, sep='\t', names=rating_cols, engine='python')
    test_df  = pd.read_csv(test_path,  sep='\t', names=rating_cols, engine='python')

    print(f"\nFold {fold_idx} loaded -> {len(train_df)} train ratings, {len(test_df)} test ratings")

    return train_df, test_df


def preprocess_users(users_df):
    print("Preprocessing user data...")
    scaler = StandardScaler()

    # 0 for female, 1 for male
    users_df['gender'] = users_df['gender'].map({'F': 0, 'M': 1})

    # Convert occupation to categorical integer type
    users_df['occupation'] = pd.Categorical(users_df['occupation']).codes

    users_df[['age', 'occupation']] = scaler.fit_transform(users_df[['age', 'occupation']])

    processed_users_df = users_df[['user_id', 'age', 'gender', 'occupation']]

    print("User data preprocessed.")
    return processed_users_df


def preprocess_items(items_df):
    print("Preprocessing item data...")
    scaler = StandardScaler()

    # Extract release year from 'release_date'
    items_df['release_date'] = pd.to_datetime(items_df['release_date'], errors='coerce')
    items_df['release_year'] = items_df['release_date'].dt.year
    items_df['release_year'] = items_df['release_year'].fillna(0).astype(int)

    items_df['release_year'] = scaler.fit_transform(items_df[['release_year']])


    genre_columns = [f'genre_{i}' for i in range(19)]
    movie_features = ['movie_id', 'title', 'release_year'] + genre_columns

    # Select relevant columns
    processed_items_df = items_df[movie_features]

    print("Item data preprocessed.")
    return processed_items_df   



def create_heterodata(users_df, items_df, ratings_df):
    print("\nCreating HeteroData object...")

    data = HeteroData()

    # --- Add Nodes ---
    user_ids = users_df['user_id'].unique()
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    num_users = len(user_ids)
    
    # user features: age, gender, occupation
    user_features_tensor = torch.tensor(users_df[['age', 'gender', 'occupation']].values, dtype=torch.float)

    data['user'].x = user_features_tensor
    data['user'].num_nodes = num_users
    
    print(f"Added {num_users} user nodes with features.")

    # Add movie nodes
    movie_ids = items_df['movie_id'].unique()
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(movie_ids)}
    num_movies = len(movie_ids)

    # movie features: release_year and genre information
    genre_cols_to_use = [f'genre_{i}' for i in range(19)]

    cols = ['release_year'] + genre_cols_to_use
    item_features_tensor = torch.tensor(items_df[cols].values, dtype=torch.float)

    data['movie'].x = item_features_tensor
    data['movie'].num_nodes = num_movies

    print(f"Added {num_movies} movie nodes with features.")


    # --- Add Edges (Interactions) ---
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_id_map)
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_id_map)

    ratings_df = ratings_df.dropna(subset=['user_idx', 'movie_idx'])
    ratings_df['user_idx'] = ratings_df['user_idx'].astype(int)
    ratings_df['movie_idx'] = ratings_df['movie_idx'].astype(int)

    # Create edge index for user -> movie interactions
    edge_index = torch.tensor(ratings_df[['user_idx', 'movie_idx']].values, dtype=torch.long).t().contiguous()
    
    # Add edge features (the rating itself)
    edge_features = torch.tensor(ratings_df['rating'].values, dtype=torch.float).view(-1, 1)

    # Add the edge data to the HeteroData object
    data['user', 'rates', 'movie'].edge_index = edge_index
    data['user', 'rates', 'movie'].edge_attr = edge_features # Store rating as edge attribute

    print(f"Added {edge_index.size(1)} user-movie rating edges.")

    # Add reverse edges 
    edge_index_rev = torch.flip(edge_index, dims=[0])
    data['movie', 'rated_by', 'user'].edge_index = edge_index_rev
    data['movie', 'rated_by', 'user'].edge_attr = edge_features # Same attributes for reverse edges

    print("HeteroData object created successfully.")
    return data



if __name__ == '__main__':
    current_dir = os.getcwd()
    data_directory = os.path.join(current_dir, "../Data/ml-100k/") 

    print(f"Attempting to load data from: {os.path.abspath(data_directory)}")
    
    if not os.path.exists(data_directory):
        print(f"Error: Data directory not found at {os.path.abspath(data_directory)}")
        print("Please ensure the 'Data/ml-100k/' directory exists and contains the dataset files (u.user, u.item, u.data).")
    else:
        # Load raw data
        users_df, items_df, ratings_df = load_ml_100k_data(data_dir=data_directory)

        # Preprocess data
        processed_users_df = preprocess_users(users_df.copy()) # Use .copy() to avoid SettingWithCopyWarning
        processed_items_df = preprocess_items(items_df.copy()) # Use .copy()

        print(processed_users_df)
        print(processed_items_df)

        # Load one predefined fold
        train_df, test_df = load_ml_100k_fold(data_directory, fold_idx=1)

        # Build graph objects for train/test sets
        hetero_train = create_heterodata(processed_users_df, processed_items_df, train_df)
        hetero_test = create_heterodata(processed_users_df, processed_items_df, test_df)

        # Create HeteroData object
        # hetero_data = create_heterodata(processed_users_df, processed_items_df, ratings_df)

        print("\n--- HeteroData Summary ---")
        print(hetero_test)
        print(f"Number of user nodes: {hetero_test['user'].num_nodes}")
        print(f"Shape of user features: {hetero_test['user'].x.shape}")
        print(f"Number of movie nodes: {hetero_test['movie'].num_nodes}")
        print(f"Shape of movie features: {hetero_test['movie'].x.shape}")
        print(f"Number of rating edges: {hetero_test['user', 'rates', 'movie'].edge_index.size(1)}")
        print(f"Shape of rating edge attributes: {hetero_test['user', 'rates', 'movie'].edge_attr.shape}")
