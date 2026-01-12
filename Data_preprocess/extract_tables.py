from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, regexp_replace
import os


def create_spark_session(app_name="Extract Twitter Data"):
    """Create Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark


def extract_interaction_table(spark, base_dir, posts_all_path=None, posts_path=None):
    """
    Extract interaction table
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    base_dir : str
        Base directory path
    posts_all_path : str, optional
        posts_all.jsonl file path, if None use default path
    posts_path : str, optional
        posts.jsonl file path, if None use default path
        
    Returns:
    --------
    DataFrame
        Interaction table data
    """
    if posts_path is None:
        posts_path = os.path.join(base_dir, "posts.jsonl")
    if posts_all_path is None:
        posts_all_path = os.path.join(base_dir, "posts_all.jsonl")
    
    # Check if file exists
    if not os.path.exists(posts_path):
        raise FileNotFoundError(f"File not found: {posts_path}")
    
    # Read data
    df_full = spark.read.json(posts_path, multiLine=False)
    
    # If sample file exists, read and merge
    if os.path.exists(posts_all_path):
        df_sample = spark.read.json(posts_all_path, multiLine=False)
        df = df_sample.union(df_full)
    else:
        df = df_full
    
    # Extract required fields for interaction table
    interaction_df = df.select(
        col("account_id").alias("id_i"),
        col("id").alias("msg_id"),
        col("send_time").alias("time"),
        when(col("type") == "post", lit(1))
            .when(col("type") == "comment", lit(2))
            .when(col("type") == "retweet", lit(3))
            .otherwise(lit(None)).alias("type"),
        col("parent_id").alias("id_j")
    )
    
    return interaction_df, df


def extract_message_table(spark, df):
    """
    Extract message table
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    df : DataFrame
        Original posts data
        
    Returns:
    --------
    DataFrame
        Message table data
    """
    cleaned_text_col = regexp_replace(
        regexp_replace(
            regexp_replace(col("content.text"), '\n', ' '),
            '\r', ' '
        ),
        '"', "'"
    ).alias("text")
    
    message_df = df.select(
        col("id").alias("msg_id"),
        cleaned_text_col,
        col("send_time").alias("time"),
        col("statistics.cnt_like").alias("like_count"),
        col("statistics.cnt_comment").alias("comment_count"),
        col("statistics.cnt_share").alias("share_count")
    )
    
    return message_df


def extract_user_table(spark, base_dir, accounts_path=None):
    """
    Extract user table
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    base_dir : str
        Base directory path
    accounts_path : str, optional
        accounts.jsonl file path, if None use default path
        
    Returns:
    --------
    DataFrame
        User table data
    """
    if accounts_path is None:
        accounts_path = os.path.join(base_dir, "accounts.jsonl")
    
    if not os.path.exists(accounts_path):
        raise FileNotFoundError(f"File not found: {accounts_path}")
    
    user_df = spark.read.json(accounts_path, multiLine=False)
    
    cleaned_intro_col = regexp_replace(
        regexp_replace(
            regexp_replace(col("intro"), '\n', ' '),
            '\r', ' '
        ),
        '"', "'"
    ).alias("intro")
    
    # Extract fields
    user_df = user_df.select(
        col("id").alias("id_u"),
        col("name").alias("name"),
        cleaned_intro_col,
        col("source.platform").alias("platform"),
        col("statistics.cnt_following").alias("following_count"),
        col("statistics.cnt_follower").alias("follower_count"),
        col("statistics.cnt_posts").alias("posts")
    )
    
    return user_df


def save_tables(interaction_df, message_df, user_df, base_dir):
    """
    Save three tables to CSV files
    
    Parameters:
    -----------
    interaction_df : DataFrame
        Interaction table data
    message_df : DataFrame
        Message table data
    user_df : DataFrame
        User table data
    base_dir : str
        Base directory path
    """
    # Save interaction table
    interaction_output_path = os.path.join(base_dir, "interaction")
    interaction_df.write.mode("overwrite").option("header", True).csv(interaction_output_path)
    print(f"Interaction table saved to: {interaction_output_path}")
    
    # Save message table
    message_output_path = os.path.join(base_dir, "message")
    message_df.write.mode("overwrite") \
        .option("header", True) \
        .option("quote", "\"") \
        .csv(message_output_path)
    print(f"Message table saved to: {message_output_path}")
    
    # Save user table
    user_output_path = os.path.join(base_dir, "user")
    user_df.write.mode("overwrite").option("header", True).csv(user_output_path)
    print(f"User table saved to: {user_output_path}")


def main(base_dir=None, posts_all_path=None, posts_path=None, accounts_path=None):
    """
    Main function: Extract three tables (interaction, message, user)
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory path, if None use project root directory
    posts_all_path : str, optional
        posts_all.jsonl file path
    posts_path : str, optional
        posts.jsonl file path
    accounts_path : str, optional
        accounts.jsonl file path
    """
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Determine actual file paths to use
        if posts_path is None:
            posts_path = os.path.join(base_dir, "posts.jsonl")
        if posts_all_path is None:
            posts_all_path = os.path.join(base_dir, "posts_all.jsonl")
        if accounts_path is None:
            accounts_path = os.path.join(base_dir, "accounts.jsonl")
        
        print(f"\nInput file paths:")
        print(f"   - posts.jsonl: {posts_path}")
        print(f"   - posts_all.jsonl: {posts_all_path} (if exists)")
        print(f"   - accounts.jsonl: {accounts_path}")
        
        # Extract interaction table
        print("\nExtracting interaction table...")
        interaction_df, posts_df = extract_interaction_table(spark, base_dir, posts_all_path, posts_path)
        
        # Extract message table
        print("Extracting message table...")
        message_df = extract_message_table(spark, posts_df)
        
        # Extract user table
        print("Extracting user table...")
        user_df = extract_user_table(spark, base_dir, accounts_path)
        
        # Save all tables
        print("Saving table data...")
        save_tables(interaction_df, message_df, user_df, base_dir)
        
        print("All tables extracted successfully!")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
