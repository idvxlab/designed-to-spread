from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType
from tqdm import tqdm
import json
import os


def create_spark_session(app_name="Extract Network Structure"):
    """Create Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.executor.memoryOverhead", "2048") \
        .config("spark.network.timeout", "1200s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    spark.catalog.clearCache()
    return spark


def extract_network_structure(spark, base_dir, batch_size=50000, repartition_num=4):
    """
    Extract network structure
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    base_dir : str
        Base directory path
    batch_size : int
        Batch processing size, default 50000
    repartition_num : int
        Number of repartitions, default 4
        
    Returns:
    --------
    DataFrame
        Filtered valid edge data
    """
    # Read data from output directory of 01_extract_tables.py
    interaction_df = spark.read.option("header", True).csv(f"{base_dir}/interaction")
    user_df = spark.read.option("header", True).csv(f"{base_dir}/user")

    # Cache interaction_df to avoid repeated reads
    interaction_df.cache()
    
    interaction_count = interaction_df.count()
    print(f"interaction_df data count: {interaction_count}")

    # Filter type=2 or 3
    filtered_df = interaction_df.filter(interaction_df.type.isin([2, 3]))
    filtered_df.cache()

    # Get id_j list - use distinct and limit to avoid collecting too much data
    interactions = filtered_df.select(col("id_i").alias("user_j"), col("id_j").alias("tweet_id"))
    
    # Get unique tweet_id, avoid collecting too much data
    tweet_ids_df = interactions.select("tweet_id").distinct()
    tweet_ids_df.cache()
    
    # Get count for progress bar
    tweet_count = tweet_ids_df.count()
    print(f"Number of unique tweet_id to process: {tweet_count}")
    
    # Use join directly instead of collect + loop
    # Join interaction_df with filtered_df's tweet_id
    combined_df = interaction_df.repartition(repartition_num)
    combined_df.cache()
    
    # Use join instead of collect + loop batch processing
    result_df = combined_df.join(
        tweet_ids_df.select(col("tweet_id").alias("msg_id")), 
        on="msg_id", 
        how="inner"
    ).select("msg_id", "id_i").dropDuplicates(["msg_id"])
    
    result_df.cache()
    result_count = result_df.count()
    print(f"result_df data count: {result_count}")

    # Associate user_i information
    updated_interaction_df = interactions.join(result_df, interactions.tweet_id == result_df.msg_id, how="left") \
                                         .withColumnRenamed("id_i", "user_i") \
                                         .drop("msg_id")
    updated_interaction_df.cache()

    filtered_interaction_df = updated_interaction_df.filter(col("user_i").isNotNull())
    filtered_interaction_df.cache()

    final_df = filtered_interaction_df.select("user_i", "tweet_id", "user_j") \
                                     .dropDuplicates(["user_i", "tweet_id", "user_j"])
    final_df.cache()

    final_count = final_df.count()
    print(f"Final edge count: {final_count}")

    # Check user overlap, prepare for filtering
    interaction_ids = final_df.select("user_i").union(final_df.select("user_j")) \
                             .withColumnRenamed("user_i", "user_id").distinct()
    interaction_ids.cache()
    
    user_ids_df = user_df.select(col("id_u").alias("user_id")).distinct()
    user_ids_df.cache()

    common_users_count = interaction_ids.join(user_ids_df, on="user_id", how="inner").count()
    print("Common users in both tables:", common_users_count)

    missing_users_count = interaction_ids.join(user_ids_df, on="user_id", how="left_anti").count()
    print("Users in valid_edges but missing from user table:", missing_users_count)

    only_in_user_count = user_ids_df.join(interaction_ids, on="user_id", how="left_anti").count()
    print("Users in user table not participating in interactions:", only_in_user_count)

    # Filter final_df to ensure user_i and user_j are both in user table
    # Use broadcast to optimize join for small tables
    user_ids_broadcast = broadcast(user_ids_df)
    
    filtered_valid_edges = final_df.join(user_ids_broadcast, final_df.user_i == user_ids_broadcast.user_id, how="inner") \
                                   .drop("user_id")
    filtered_valid_edges = filtered_valid_edges.join(user_ids_broadcast, filtered_valid_edges.user_j == user_ids_broadcast.user_id, how="inner") \
                                               .drop("user_id")
    filtered_valid_edges.cache()

    filtered_count = filtered_valid_edges.count()
    print(f"Final edge count after filtering: {filtered_count}")

    filtered_valid_edges.show(5)

    return filtered_valid_edges


def extract_relations_network(spark, relations_file_path, base_dir, user_df):
    """
    Extract relation network from relations_all.jsonl and generate valid_edges_relations
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    relations_file_path : str
        relations_all.jsonl file path
    base_dir : str
        Base directory path
    user_df : DataFrame
        User table data
        
    Returns:
    --------
    DataFrame
        Filtered relation edge data
    """
    if not os.path.exists(relations_file_path):
        print(f"Warning: relations_all.jsonl file not found: {relations_file_path}")
        print("Skipping relation network extraction")
        return None
    
    print("Starting to extract relation network...")
    
    # Read JSONL file
    relations_data = []
    with open(relations_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Only process follow relations
                if data.get('kind') == 'follow':
                    src_id = data.get('src_account_id', '')
                    dst_id = data.get('dst_account_id', '')
                    if src_id and dst_id:
                        relations_data.append((src_id, dst_id))
            except json.JSONDecodeError:
                continue
    
    if not relations_data:
        print("Warning: No valid follow relations found in relations_all.jsonl")
        return None
    
    print(f"Read {len(relations_data)} follow relations from relations_all.jsonl")
    
    # Create DataFrame
    schema = StructType([
        StructField("user_i", StringType(), False),
        StructField("user_j", StringType(), False)
    ])
    relations_df = spark.createDataFrame(relations_data, schema)
    relations_df.cache()
    
    # Get user ID list for filtering
    user_ids_df = user_df.select(col("id_u").alias("user_id")).distinct()
    user_ids_df.cache()
    user_ids_broadcast = broadcast(user_ids_df)
    
    # Filter: ensure user_i and user_j are both in user table
    filtered_relations = relations_df.join(
        user_ids_broadcast, 
        relations_df.user_i == user_ids_broadcast.user_id, 
        how="inner"
    ).drop("user_id")
    
    filtered_relations = filtered_relations.join(
        user_ids_broadcast, 
        filtered_relations.user_j == user_ids_broadcast.user_id, 
        how="inner"
    ).drop("user_id")
    
    filtered_relations.cache()
    filtered_count = filtered_relations.count()
    print(f"Filtered relation edge count: {filtered_count}")
    
    return filtered_relations


def save_results(df, output_path):
    """
    Save results to CSV file
    
    Parameters:
    -----------
    df : DataFrame
        Data to save
    output_path : str
        Output path
    """
    df.write.mode("overwrite").option("header", True).csv(output_path)
    print(f"Results saved to: {output_path}")


def main(base_dir=None):
    """Main function"""
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    
    # Read relations_all.jsonl from Data directory using relative path
    relations_file_path = os.path.join(base_dir, "relations_all.jsonl")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Read user table (for filtering relation network)
        user_df = spark.read.option("header", True).csv(f"{base_dir}/user")
        
        # Extract network structure
        filtered_valid_edges = extract_network_structure(spark, base_dir)
        
        # Save results
        output_path = f"{base_dir}/valid_edges"
        save_results(filtered_valid_edges, output_path)
        
        # Extract relation network
        relations_edges = extract_relations_network(spark, relations_file_path, base_dir, user_df)
        if relations_edges is not None:
            relations_output_path = f"{base_dir}/valid_edges_relations"
            save_results(relations_edges, relations_output_path)
        
    finally:
        # Close Spark session
        spark.stop()


if __name__ == "__main__":
    main()
