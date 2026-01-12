from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, array_union, col, udf
from pyspark.sql.types import StringType
import json
import random
import os
import re


def create_spark_session():
    """创建 SparkSession 并设置必要配置。"""
    spark = SparkSession.builder \
        .appName("TextDatasetSplit") \
        .config("spark.driver.memory", "16g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark


def load_data(spark, base_dir):
    """读取 02 步生成的 valid_edges 以及 01 步生成的 message 数据。"""
    edges_path = f"{base_dir}/valid_edges"
    message_path = f"{base_dir}/message"

    edges_df = spark.read.option("header", True).csv(edges_path)
    message_df = spark.read.option("header", True).csv(message_path)
    return edges_df, message_df


def prepare_filtered_edges(edges_df, message_df):
    """过滤出同时存在于消息表中的边，并返回有效 tweet_id。"""
    filtered_edges = edges_df.join(
        message_df,
        edges_df["tweet_id"] == message_df["msg_id"],
        "inner"
    ).select(edges_df["*"]).persist()

    valid_msg_ids_df = filtered_edges.select("tweet_id").distinct() \
        .withColumnRenamed("tweet_id", "msg_id")

    return filtered_edges, valid_msg_ids_df


def build_clean_text_udf():
    """构建文本清洗 UDF。"""
    def clean_text(message):
        if not isinstance(message, str):
            return ''
        message = re.sub(r'http\S+|www.\S+', ' ', message)
        message = re.sub(r'@\w+', ' ', message)
        message = re.sub(r'#\w+', ' ', message)
        message = re.sub(r'[\'\"“”‘’]', ' ', message)
        message = re.sub(r'\s+', ' ', message).strip()
        return message if message else ' '

    return udf(clean_text, StringType())


def group_text_messages(message_df, valid_msg_ids_df, clean_text_udf):
    """按清洗后的文本分组汇总 msg_id。"""
    text_msgids_df = message_df.select("msg_id", "text") \
        .dropna().dropDuplicates() \
        .join(valid_msg_ids_df, on="msg_id", how="inner") \
        .withColumn("cleaned_text", clean_text_udf(col("text")))

    return text_msgids_df.groupBy("cleaned_text") \
        .agg(collect_list("msg_id").alias("msg_ids"))


def build_tweet_user_dict(spark, filtered_edges):
    """构建 tweet_id -> user_set 映射（用于划分时检查用户重合）。"""
    tweet_user_df = filtered_edges.groupBy("tweet_id") \
        .agg(
            collect_list(col("user_i").cast("string")).alias("user_i_list"),
            collect_list(col("user_j").cast("string")).alias("user_j_list")
        ) \
        .withColumn("all_users", array_union(col("user_i_list"), col("user_j_list"))) \
        .select("tweet_id", "all_users")

    return {
        str(row["tweet_id"]): set(row["all_users"])
        for row in tweet_user_df.toLocalIterator()
    }


def get_users_from_msg_ids(msg_ids, tweet_user_dict):
    """获取多个 msg_id 的所有用户集合。"""
    return set.union(*(tweet_user_dict.get(str(mid), set()) for mid in msg_ids)) if msg_ids else set()


def split_msg_ids(text_msgids_grouped, tweet_user_dict, train_ratio=0.8, seed=40):
    """根据文本分组和用户约束切分训练/测试 msg_id 集合，确保文本和用户都不重合。"""
    text_msgids_list = text_msgids_grouped.rdd.map(
        lambda row: (row['cleaned_text'], row['msg_ids'])
    ).collect()

    random.seed(seed)
    random.shuffle(text_msgids_list)

    total_msg_count = sum(len(msg_ids) for _, msg_ids in text_msgids_list)
    MAX_TEST_SIZE = total_msg_count // 5
    success = False

    for test_target in reversed(range(1, MAX_TEST_SIZE + 1)):
        train_target = test_target * 4

        train_msg_ids, test_msg_ids = set(), set()
        train_users, test_users = set(), set()
        train_texts, test_texts = set(), set()

        remaining_text_msgids = []

        # Step 1: 选测试集（确保用户不重合）
        for text, msg_ids in text_msgids_list:
            users = get_users_from_msg_ids(msg_ids, tweet_user_dict)
            if users.isdisjoint(test_users):
                if len(test_msg_ids) + len(msg_ids) <= test_target:
                    test_msg_ids.update(msg_ids)
                    test_users.update(users)
                    test_texts.add(text)
                else:
                    remaining_text_msgids.append((text, msg_ids))
            else:
                remaining_text_msgids.append((text, msg_ids))

        # Step 2: 选训练集（确保用户与训练集和测试集都不重合）
        for text, msg_ids in remaining_text_msgids:
            users = get_users_from_msg_ids(msg_ids, tweet_user_dict)
            if users.isdisjoint(train_users) and users.isdisjoint(test_users):
                if len(train_msg_ids) + len(msg_ids) <= train_target:
                    train_msg_ids.update(msg_ids)
                    train_users.update(users)
                    train_texts.add(text)
                if len(train_msg_ids) >= train_target:
                    break

        if len(train_msg_ids) >= train_target and len(test_msg_ids) >= test_target:
            print(f"\n✅ Split successful! test tweets: {len(test_msg_ids)}, train tweets: {len(train_msg_ids)}")
            success = True
            break

    if not success:
        raise ValueError("❌ Split failed; please add more data or relax constraints.")

    return train_msg_ids, test_msg_ids


def split_edges(spark, filtered_edges, train_msg_ids, test_msg_ids):
    """根据 msg_id 集合划分边。"""
    train_ids_df = spark.createDataFrame([(i,) for i in train_msg_ids], ["msg_id"])
    test_ids_df = spark.createDataFrame([(i,) for i in test_msg_ids], ["msg_id"])

    train_edges = filtered_edges.join(
        train_ids_df, filtered_edges["tweet_id"] == train_ids_df["msg_id"], "inner"
    ).drop("msg_id")
    test_edges = filtered_edges.join(
        test_ids_df, filtered_edges["tweet_id"] == test_ids_df["msg_id"], "inner"
    ).drop("msg_id")

    return train_edges, test_edges


def build_tweet_user_mapping(edges_df):
    """构建 tweet_id -> 用户列表 映射。"""
    grouped = edges_df.groupBy("tweet_id") \
        .agg(
            collect_list(col("user_i").cast("string")).alias("user_i_list"),
            collect_list(col("user_j").cast("string")).alias("user_j_list")
        ) \
        .withColumn("all_users", array_union(col("user_i_list"), col("user_j_list"))) \
        .select("tweet_id", "all_users")

    return {
        str(row['tweet_id']): ",".join(sorted(set(row['all_users'])))
        for row in grouped.toLocalIterator()
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def summarize_user_stats(train_dict, test_dict):
    """打印用户统计信息。"""
    train_users = set()
    for users in train_dict.values():
        train_users.update(users.split(','))

    test_users = set()
    for users in test_dict.values():
        test_users.update(users.split(','))

    common_users = train_users & test_users

    print(f"\nTrain user count: {len(train_users)}")
    print(f"Test user count: {len(test_users)}")
    print(f"Common users: {len(common_users)}")
    if test_users:
        print(f"User overlap rate (based on test set): {len(common_users) / len(test_users):.2%}")
    else:
        print("Test user count is 0; cannot compute overlap rate.")


def check_empty_users(data_dict, dict_name):
    """检查映射中是否存在无用户的 tweet。"""
    empty = [tid for tid, users in data_dict.items() if not users.strip()]
    if empty:
        print(f"⚠️ {dict_name} has {len(empty)} tweet_ids with empty users, examples: {empty[:5]}")
    else:
        print(f"✅ All tweet_ids in {dict_name} have users")


def main(base_dir=None):
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    output_dir = os.path.join(base_dir, "text_split")
    os.makedirs(output_dir, exist_ok=True)

    spark = create_spark_session()

    try:
        edges_df, message_df = load_data(spark, base_dir)
        filtered_edges, valid_msg_ids_df = prepare_filtered_edges(edges_df, message_df)

        clean_text_udf = build_clean_text_udf()
        text_msgids_grouped = group_text_messages(message_df, valid_msg_ids_df, clean_text_udf)

        # 构建 tweet_id -> user_set 映射
        tweet_user_dict = build_tweet_user_dict(spark, filtered_edges)

        # 划分训练/测试集（确保文本和用户都不重合）
        train_msg_ids, test_msg_ids = split_msg_ids(
            text_msgids_grouped, tweet_user_dict, train_ratio=0.8, seed=40
        )

        print(f"\nTrain tweet_id count: {len(train_msg_ids)}")
        print(f"Test tweet_id count: {len(test_msg_ids)}")
        total = len(train_msg_ids) + len(test_msg_ids)
        if total:
            print(f"Ratio: train = {len(train_msg_ids) / total:.2%}")
        print(f"Text overlap count (should be 0): {len(train_msg_ids & test_msg_ids)}")

        train_edges, test_edges = split_edges(spark, filtered_edges, train_msg_ids, test_msg_ids)

        train_dict = build_tweet_user_mapping(train_edges)
        test_dict = build_tweet_user_mapping(test_edges)
        all_dict = {**train_dict, **test_dict}

        save_json(all_dict, os.path.join(output_dir, "tweet_user_mapping_all.json"))
        save_json(train_dict, os.path.join(output_dir, "tweet_user_mapping_train.json"))
        save_json(test_dict, os.path.join(output_dir, "tweet_user_mapping_test.json"))

        summarize_user_stats(train_dict, test_dict)
        check_empty_users(all_dict, "All")
        check_empty_users(train_dict, "Train")
        check_empty_users(test_dict, "Test")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
