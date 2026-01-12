from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import torch
from tqdm import tqdm
import numpy as np
import os
import glob


def create_spark_session(app_name="ComputeUserFeatures"):
    """Create Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "32g") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.memoryOverhead", "1024") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.driver.maxResultSize", "16g") \
        .config("spark.sql.shuffle.partitions", "500") \
        .getOrCreate()
    
    spark.catalog.clearCache()
    return spark


def load_data(spark, base_dir):
    """
    Load required data
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    base_dir : str
        Base directory path
        
    Returns:
    --------
    tuple
        (valid_edges_df, valid_edges_relations_df, filtered_tweet_df, msg_id_features_dict)
    """
    # Read edge data from output of 02_extract_network.py
    valid_edges_df = spark.read.option("header", True).csv(f"{base_dir}/valid_edges")
    
    # Try to read relation edge data (if exists)
    valid_edges_relations_df = None
    relations_path = f"{base_dir}/valid_edges_relations"
    if os.path.exists(relations_path):
        try:
            valid_edges_relations_df = spark.read.option("header", True).csv(relations_path)
            print(f"Successfully loaded relation edge data: {valid_edges_relations_df.count()} edges")
        except Exception as e:
            print(f"Warning:  Failed to read relation edge data: {e}")
    else:
        print("  valid_edges_relations not found, will only use valid_edges")
    
    # Read filtered_tweet_data from output of 05_filter_texts.py
    filtered_tweet_dir = os.path.join(base_dir, "features", "filtered_tweet_data")
    csv_files = sorted([
        f for f in glob.glob(os.path.join(filtered_tweet_dir, "*.csv"))
        if "part-" in os.path.basename(f)
    ])
    
    if not csv_files:
        raise FileNotFoundError(f"filtered_tweet_data file not found: {filtered_tweet_dir}")
    
    # Read all part files and merge
    all_dfs = []
    for csv_file in csv_files:
        df = spark.read.option("header", True).csv(csv_file)
        all_dfs.append(df)
    
    if len(all_dfs) == 1:
        filtered_tweet_df = all_dfs[0]
    else:
        from functools import reduce
        from pyspark.sql import DataFrame
        filtered_tweet_df = reduce(DataFrame.unionByName, all_dfs)
    
    # Read message feature dictionary from output of 06_compute_message_features.py
    msg_features_path = os.path.join(base_dir, "features", "msg_id_features_dict.pt")
    if not os.path.exists(msg_features_path):
        raise FileNotFoundError(f"Message feature file not found: {msg_features_path}")
    
    msg_id_features_dict = torch.load(msg_features_path)
    
    print(f"Data loaded successfully!")
    print(f"   Edge data rows: {valid_edges_df.count()}")
    if valid_edges_relations_df is not None:
        print(f"   Relation edge data rows: {valid_edges_relations_df.count()}")
    print(f"   User post data rows: {filtered_tweet_df.count()}")
    print(f"   Message feature count: {len(msg_id_features_dict)}")
    
    return valid_edges_df, valid_edges_relations_df, filtered_tweet_df, msg_id_features_dict


def get_user_list(valid_edges_df, valid_edges_relations_df=None):
    """
    Extract all user IDs from edge data (from valid_edges and valid_edges_relations, deduplicated)
    
    Parameters:
    -----------
    valid_edges_df : DataFrame
        Edge data
    valid_edges_relations_df : DataFrame, optional
        Relation edge data
        
    Returns:
    --------
    DataFrame
        DataFrame containing all unique user IDs
    """
    # Extract user IDs from valid_edges
    user_ids_from_edges = valid_edges_df.select("user_i").union(
        valid_edges_df.select("user_j")
    ).distinct()
    
    # If relation edge data exists, also extract user IDs
    if valid_edges_relations_df is not None:
        user_ids_from_relations = valid_edges_relations_df.select("user_i").union(
            valid_edges_relations_df.select("user_j")
        ).distinct()
        # Merge and deduplicate
        user_ids_df = user_ids_from_edges.union(user_ids_from_relations).distinct()
    else:
        user_ids_df = user_ids_from_edges
    
    # Rename column to user
    user_ids_df = user_ids_df.withColumnRenamed("user_i", "user")
    
    total_users = user_ids_df.count()
    print(f"Unique user count in edge data (after deduplication): {total_users}")
    if valid_edges_relations_df is not None:
        print(f"   - From valid_edges: {user_ids_from_edges.count()}")
        print(f"   - From valid_edges_relations: {valid_edges_relations_df.select('user_i').union(valid_edges_relations_df.select('user_j')).distinct().count()}")
    
    return user_ids_df


def prepare_user_tweet_mapping(filtered_tweet_df, user_ids_df):
    """
    Prepare user-message mapping relationship
    
    Parameters:
    -----------
    filtered_tweet_df : DataFrame
        User post data
    user_ids_df : DataFrame
        User ID list
        
    Returns:
    --------
    list
        [(user, [msg_id1, msg_id2, ...]), ...] format list
    """
    # Rename id_i in filtered_tweet_df to user for easy join
    user_tweet_df = filtered_tweet_df.select(
        col('id_i').alias('user'),
        col('msg_id')
    )
    
    # Filter to only include post data from users we care about
    user_tweet_filtered_df = user_tweet_df.join(user_ids_df, on='user', how='inner')
    
    # Collect all user (user, msg_id) mappings to driver
    user_msg_pairs = user_tweet_filtered_df.select('user', 'msg_id').rdd.map(
        lambda row: (row['user'], row['msg_id'])
    ).groupByKey().mapValues(list).collect()
    
    print(f" Number of users with message mappings: {len(user_msg_pairs)}")
    
    return user_msg_pairs


def compute_user_features(user_msg_pairs, msg_id_features_dict):
    """
    Compute user features (average features of all user messages)
    
    Parameters:
    -----------
    user_msg_pairs : list
        [(user, [msg_id1, msg_id2, ...]), ...] format list
    msg_id_features_dict : dict
        Dictionary from message ID to features {msg_id: tensor}
        
    Returns:
    --------
    dict
        Dictionary from user ID to features {user: tensor}
    """
    user_features_dict = {}
    users_without_features = 0
    
    for user, msg_id_list in tqdm(user_msg_pairs, desc="Computing user features"):
        feature_list = []
        for msg_id in msg_id_list:
            # Try different msg_id formats (string or number)
            feature = msg_id_features_dict.get(str(msg_id), None)
            if feature is None:
                # If string format not found, try number format
                try:
                    feature = msg_id_features_dict.get(int(msg_id), None)
                except (ValueError, TypeError):
                    pass
            
            if feature is not None:
                # If tensor, convert to numpy
                if isinstance(feature, torch.Tensor):
                    feature_list.append(feature.numpy())
                else:
                    feature_list.append(np.array(feature))
        
        if feature_list:
            # Average all message features
            mean_feature = np.mean(feature_list, axis=0)
            user_features_dict[user] = torch.tensor(mean_feature, dtype=torch.float32)
        else:
            users_without_features += 1
    
    if users_without_features > 0:
        print(f"Warning: {users_without_features} users did not find corresponding message features")
    
    print(f" Successfully computed {len(user_features_dict)} user features")
    
    return user_features_dict


def save_user_features(user_features_dict, output_path):
    """
    Save user feature dictionary
    
    Parameters:
    -----------
    user_features_dict : dict
        User feature dictionary
    output_path : str
        Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(user_features_dict, output_path)
    print(f" User feature dictionarySaved to: {output_path}")
    print(f"   User count: {len(user_features_dict)}")
    if user_features_dict:
        sample_user = list(user_features_dict.keys())[0]
        sample_feature = user_features_dict[sample_user]
        print(f"   Feature dimension: {sample_feature.shape}")


def main(base_dir=None):
    """Main function"""
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    output_path = os.path.join(base_dir, "features", "user_features_dict.pt")
    
    spark = create_spark_session()
    
    try:
        # Load data
        valid_edges_df, valid_edges_relations_df, filtered_tweet_df, msg_id_features_dict = load_data(spark, base_dir)
        
        # Get user list (extracted from valid_edges and valid_edges_relations, deduplicated)
        user_ids_df = get_user_list(valid_edges_df, valid_edges_relations_df)
        
        # Prepare user-message mapping
        user_msg_pairs = prepare_user_tweet_mapping(filtered_tweet_df, user_ids_df)
        
        # Computing user features
        user_features_dict = compute_user_features(user_msg_pairs, msg_id_features_dict)
        
        # Save results
        save_user_features(user_features_dict, output_path)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
