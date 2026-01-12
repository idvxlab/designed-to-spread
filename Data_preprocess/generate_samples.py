from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, expr, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import json
import os
import random


def create_spark_session(app_name="SampleGeneration"):
    """Create Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.catalog.clearCache()
    return spark


def load_json_dict(path):
    """Load JSON dictionary file"""
    with open(path, "r") as f:
        return json.load(f)


def dict_to_spark_df(spark, d):
    """Convert dictionary to Spark DataFrame"""
    rows = [(k, v.split(',')) for k, v in d.items()]
    schema = StructType([
        StructField("tweet_id", StringType(), False),
        StructField("users",  ArrayType(StringType()), False)
    ])
    return spark.createDataFrame(rows, schema)

def define_udfs(broadcast_all_users):
    from pyspark.sql.functions import udf

    FIXED_SEED_POS_TYPE2 = 42

    def gen_pos_type1(users):
        poster = users[0]
        reusers = users[1:]
        return [[poster, u] for u in reusers] if reusers else [[poster, poster]]

    def gen_pos_type2(users):
        reusers = users[1:]
        n = len(reusers)
        if n < 2:
            return []
        pos1_count = n
        pairs = set()
        rnd = random.Random(FIXED_SEED_POS_TYPE2) 
        while len(pairs) < pos1_count:
            u1, u2 = rnd.sample(reusers, 2)
            pairs.add((u1, u2))
        return [list(p) for p in pairs]

    def gen_neg_samples(users, tweet_id, num_neg):
        all_users = broadcast_all_users.value
        tweet_users = set(users)
        neg_users = [u for u in all_users if u not in tweet_users]
        if len(neg_users) < 2:
            return []
        # 用 tweet_id 的 hash + 固定种子确保每个 tweet_id 负样本固定
        seed_for_neg = 1000 + abs(hash(tweet_id)) % (2**32 - 1)
        rnd = random.Random(seed_for_neg)
        pairs = set()
        while len(pairs) < num_neg:
            u1, u2 = rnd.sample(neg_users, 2)
            pairs.add((u1, u2))
        return [list(p) for p in pairs]

    return (
        udf(gen_pos_type1, ArrayType(ArrayType(StringType()))),
        udf(gen_pos_type2, ArrayType(ArrayType(StringType()))),
        gen_neg_samples  # Note: not a udf
    )

from pyspark.sql.functions import count, collect_list, struct, rand, monotonically_increasing_id
from pyspark.sql import Row

def generate_samples(spark, df, udf_pos_type1, udf_pos_type2, gen_neg_samples_fn, broadcast_all_users, sample_type):
    # Step 1: Generate original positive samples
    pos_df = df \
        .withColumn("pos_type1", udf_pos_type1(col("users"))) \
        .withColumn("pos_type2", udf_pos_type2(col("users"))) \
        .withColumn("pos_samples", expr("pos_type1 || pos_type2")) \
        .select("tweet_id", "users", "pos_samples")

    # Expand to each edge
    exploded_pos_df = pos_df.select(
        "tweet_id", "users", explode("pos_samples").alias("edge")
    ).select(
        col("tweet_id"),
        col("users"),
        col("edge").getItem(0).alias("user_i"),
        col("edge").getItem(1).alias("user_j")
    )

    # 第二步：每个tweet_id进rows重采样到原始2倍（have放回采样）
    def resample_pos(values_with_users):
        if not values_with_users:
            return []
        users = values_with_users[0][0]
        edges = [v[1] for v in values_with_users]
        n = len(edges)
        seed_resample = 2000 + abs(hash(users[0])) % (2**32 - 1)  # 用发帖人users[0] hash + 常数确定随机种子
        rnd = random.Random(seed_resample)
        sampled_edges = [rnd.choice(edges) for _ in range(2 * n)]
        return [(users, e) for e in sampled_edges]

    def row_from_tweet_and_edge(tweet_id, users, edge):
        return Row(user_i=edge[0], tweet_id=tweet_id, user_j=edge[1], users=users)

    grouped_pos_rdd = (
        exploded_pos_df.rdd
        .map(lambda row: (row['tweet_id'], (row['users'], (row['user_i'], row['user_j']))))
        .groupByKey()
        .mapValues(lambda values: resample_pos(list(values)))
    )

    resampled_pos_rdd = grouped_pos_rdd.flatMap(
        lambda x: [row_from_tweet_and_edge(x[0], users, edge) for users, edge in x[1]]
    )
    resampled_pos_df = spark.createDataFrame(resampled_pos_rdd).drop("users")

    # Step 3: Generate equal number of negative samples based on positive sample count for each tweet_id
    # Recalculate negative sample count (2x original positive sample count)
    pos_count_df = pos_df.select("tweet_id", "users", "pos_samples") \
        .rdd.map(lambda row: (row["tweet_id"], (row["users"], len(row["pos_samples"])))) \
        .map(lambda x: (x[0], x[1][0], 2 * x[1][1]))  # 2倍采样

    # Generate negative samples
    def gen_neg_for_row(row):
        tweet_id, users, num_neg = row
        return [(u1, tweet_id, u2) for u1, u2 in gen_neg_samples_fn(users, tweet_id, num_neg)]

    neg_rdd = pos_count_df.flatMap(gen_neg_for_row)

    neg_df = spark.createDataFrame(neg_rdd, ["user_i", "tweet_id", "user_j"])

    # Output statistics
    print(f"[{sample_type}] Positive sample count: {resampled_pos_df.count()}, Deduplicated tweet_id count: {resampled_pos_df.select('tweet_id').distinct().count()}")
    print(f"[{sample_type}] Negative sample count: {neg_df.count()}, Deduplicated tweet_id count: {neg_df.select('tweet_id').distinct().count()}")

    return resampled_pos_df, neg_df

def save_samples(pos_df, neg_df, output_dir, prefix):
    """Save positive and negative samples to CSV files"""
    pos_df.write.csv(os.path.join(output_dir, f"positive_samples_{prefix}"), header=True, mode="overwrite")
    neg_df.write.csv(os.path.join(output_dir, f"negative_samples_{prefix}"), header=True, mode="overwrite")
    print(f"Saved {prefix} positive and negative samples to: {output_dir}")


def extract_and_save_users(mapping_dict, output_path):
    """
    Extract all user IDs from mapping dictionary, deduplicate and save as JSON file
    
    Parameters:
    -----------
    mapping_dict : dict
        tweet_user_mapping 字典，格式为 {tweet_id: "user1,user2,user3"}
    output_path : str
        输出JSON文件路径
    """
    users_set = set()
    for users_str in mapping_dict.values():
        users_set.update(users_str.split(','))
    
    # 转换为排序后的列表（保持一致性）
    users_list = sorted(list(users_set))
    
    # 保存为JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(users_list, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(users_list)} user IDs to: {output_path}")


def generate_all_samples(spark, base_dir):
    """
    Generate all samples (all/train/test)
    
    Parameters:
    -----------
    spark : SparkSession
        Spark session object
    base_dir : str
        Base directory path
    """
    # 从 03基于text数据集划分.py 的Output directory读取数据
    text_split_dir = os.path.join(base_dir, "text_split")
    output_dir = os.path.join(base_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)
    
    all_dict = load_json_dict(os.path.join(text_split_dir, "tweet_user_mapping_all.json"))
    train_dict = load_json_dict(os.path.join(text_split_dir, "tweet_user_mapping_train.json"))
    test_dict = load_json_dict(os.path.join(text_split_dir, "tweet_user_mapping_test.json"))

    # 提取并保存User ID list
    print("Extract user IDs and save...")
    extract_and_save_users(all_dict, os.path.join(text_split_dir, "all_users.json"))
    extract_and_save_users(train_dict, os.path.join(text_split_dir, "train_users.json"))
    extract_and_save_users(test_dict, os.path.join(text_split_dir, "test_users.json"))
    print()

    all_df = dict_to_spark_df(spark, all_dict)
    train_df = dict_to_spark_df(spark, train_dict)
    test_df = dict_to_spark_df(spark, test_dict)

    # 收集所have用户
    all_users_set = set()
    for users_str in all_dict.values():
        all_users_set.update(users_str.split(','))
    all_users_list = list(all_users_set)
    broadcast_all_users = spark.sparkContext.broadcast(all_users_list)

    udf_pos_type1, udf_pos_type2, gen_neg_samples_fn = define_udfs(broadcast_all_users)

    # 生成所have样本
    pos_all, neg_all = generate_samples(spark, all_df, udf_pos_type1, udf_pos_type2, gen_neg_samples_fn, broadcast_all_users, "all")
    save_samples(pos_all, neg_all, output_dir, "all")

    pos_train, neg_train = generate_samples(spark, train_df, udf_pos_type1, udf_pos_type2, gen_neg_samples_fn, broadcast_all_users, "train")
    save_samples(pos_train, neg_train, output_dir, "train")

    pos_test, neg_test = generate_samples(spark, test_df, udf_pos_type1, udf_pos_type2, gen_neg_samples_fn, broadcast_all_users, "test")
    save_samples(pos_test, neg_test, output_dir, "test")


def main(base_dir=None):
    """Main function"""
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    
    spark = create_spark_session()
    
    try:
        generate_all_samples(spark, base_dir)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
