from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark import StorageLevel
import os


def create_spark_session(app_name="FilterTweetAndText"):
    """Create Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
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
        (edges_df, edges_relations_df, interaction_df, message_df)
    """
    # Read edge data from output of 02_extract_network.py
    edges_df = spark.read.option("header", True).csv(f"{base_dir}/valid_edges")
    
    # Try to read relation edge data (if exists)
    edges_relations_df = None
    relations_path = f"{base_dir}/valid_edges_relations"
    if os.path.exists(relations_path):
        try:
            edges_relations_df = spark.read.option("header", True).csv(relations_path)
            print(f" Successfully loaded relation edge data: {edges_relations_df.count()} rows")
        except Exception as e:
            print(f"Warning:  Failed to read relation edge data: {e}")
    else:
        print("  valid_edges_relations not found, will only use valid_edges")
    
    # Read interaction and message data from output of 01_extract_tables.py
    interaction_df = spark.read.option("header", True).csv(f"{base_dir}/interaction")
    message_df = spark.read.option("header", True).csv(f"{base_dir}/message")
    
    return edges_df, edges_relations_df, interaction_df, message_df

def filter_tweets_by_users(spark, edges_df, edges_relations_df, interaction_df):
    """
    Filter interaction data by users in edge data to get related tweets
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    edges_df : DataFrame
        Edge data
    edges_relations_df : DataFrame
        关系Edge data（可选）
    interaction_df : DataFrame
        Interaction data
        
    Returns:
    --------
    DataFrame
        Filtered tweet data
    """
    # Get involved user IDs (from valid_edges and valid_edges_relations)
    id_i = edges_df.select("user_i").distinct()
    id_j = edges_df.select("user_j").distinct()
    user_ids_df = id_i.union(id_j).distinct()
    
    # 如果存在关系Edge data，也加入用户ID
    if edges_relations_df is not None:
        rel_id_i = edges_relations_df.select("user_i").distinct()
        rel_id_j = edges_relations_df.select("user_j").distinct()
        rel_user_ids_df = rel_id_i.union(rel_id_j).distinct()
        user_ids_df = user_ids_df.union(rel_user_ids_df).distinct()
    
    user_ids = user_ids_df.rdd.flatMap(lambda x: x).collect()
    broadcast_user_ids = spark.sparkContext.broadcast(set(user_ids))
    
    print(f" Total number of users for filtering (after deduplication): {len(user_ids)}")
    
    # Use broadcast join for filtering
    broadcast_user_df = spark.createDataFrame([(x,) for x in broadcast_user_ids.value], ["id_i"])
    filtered_tweet_df = interaction_df.join(broadcast_user_df, on="id_i", how="inner") \
        .select("id_i", "msg_id").dropDuplicates(["id_i", "msg_id"]) \
        .repartition(10)
    
    return filtered_tweet_df

def filter_texts_by_tweets(edges_df, filtered_tweet_df, message_df):
    """
    Filter text data by tweet ID
    Includes:
    1. All tweet_id in valid_edges
    2. All posts by participating users (extracted from valid_edges and valid_edges_relations) in interaction table
    
    Parameters:
    -----------
    edges_df : DataFrame
        Edge data
    filtered_tweet_df : DataFrame
        Filtered tweet data（包含所have参与用户的发帖）
    message_df : DataFrame
        Message data
        
    Returns:
    --------
    DataFrame
        Filtered text data
    """
    # 1. Get tweet_id from valid_edges
    tweet_ids_from_edges = edges_df.select("tweet_id").distinct().withColumnRenamed("tweet_id", "msg_id")
    
    # 2. Get msg_id of all posts by participating users from filtered_tweet_df
    tweet_ids_from_users = filtered_tweet_df.select("msg_id").distinct()
    
    # 3. Merge and deduplicate
    all_tweet_ids_df = tweet_ids_from_edges.union(tweet_ids_from_users).distinct()
    
    tweet_count = all_tweet_ids_df.count()
    print(f" Total number of msg_id for filtering (after deduplication): {tweet_count}")
    print(f"   - From valid_edges: {tweet_ids_from_edges.count()}")
    print(f"   - From participating user posts: {tweet_ids_from_users.count()}")
    
    # Filter text
    texts_filtered_df = message_df.join(all_tweet_ids_df, on="msg_id", how="inner") \
        .select("msg_id", "text").dropDuplicates(["msg_id"])
    
    return texts_filtered_df

def validate_tweet_text_consistency(spark, filtered_tweet_df, texts_filtered_df):
    """
    Check if filtered_tweet_data and filtered_texts_data are consistent
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    filtered_tweet_df : DataFrame
        Filtered tweet data
    texts_filtered_df : DataFrame
        Filtered text data
    """
    tweet_msg_ids = filtered_tweet_df.select("msg_id").distinct()
    text_msg_ids = texts_filtered_df.select("msg_id").distinct()
    
    only_in_tweets = tweet_msg_ids.subtract(text_msg_ids)
    only_in_texts = text_msg_ids.subtract(tweet_msg_ids)
    
    tweet_count = tweet_msg_ids.count()
    text_count = text_msg_ids.count()
    only_tweets_count = only_in_tweets.count()
    only_texts_count = only_in_texts.count()
    
    print(f"Total msg_id count in Tweets file: {tweet_count}")
    print(f"Total msg_id count in Texts file: {text_count}")
    print(f"msg_id only in Tweets: {only_tweets_count}")
    print(f"msg_id only in Texts: {only_texts_count}")
    
    if only_tweets_count == 0 and only_texts_count == 0:
        print(" msg_id in both files are completely consistent")
    else:
        if only_tweets_count > 0:
            print("Warning: Examples only in Tweets:", only_in_tweets.limit(5).toPandas()["msg_id"].tolist())
        if only_texts_count > 0:
            print("Warning: Examples only in Texts:", only_in_texts.limit(5).toPandas()["msg_id"].tolist())


def save_results(filtered_tweet_df, texts_filtered_df, output_dir):
    """
    Save results to CSV file
    
    Parameters:
    -----------
    filtered_tweet_df : DataFrame
        Filtered tweet data
    texts_filtered_df : DataFrame
        Filtered text data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filtered_tweet_df.write.option("header", True).mode("overwrite") \
        .csv(os.path.join(output_dir, "filtered_tweet_data"))
    
    texts_filtered_df.write.mode("overwrite") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .csv(os.path.join(output_dir, "filtered_texts_data"), header=True)
    
    print(f"Results saved to: {output_dir}")


def main(base_dir=None):
    """Main function"""
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    output_dir = os.path.join(base_dir, "features")
    
    spark = create_spark_session()
    
    try:
        # 加载数据
        edges_df, edges_relations_df, interaction_df, message_df = load_data(spark, base_dir)
        
        # Filter tweets (using user IDs from valid_edges and valid_edges_relations)
        filtered_tweet_df = filter_tweets_by_users(spark, edges_df, edges_relations_df, interaction_df)
        
        # Filter text（只使用 valid_edges 中的 tweet_id）
        texts_filtered_df = filter_texts_by_tweets(edges_df, filtered_tweet_df, message_df)
        
        # Validate consistency
        validate_tweet_text_consistency(spark, filtered_tweet_df, texts_filtered_df)
        
        # Save results
        save_results(filtered_tweet_df, texts_filtered_df, output_dir)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
