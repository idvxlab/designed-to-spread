import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import os
from tqdm import tqdm


def create_spark_session():
    """创建 SparkSession 并设置必要配置。"""
    spark = SparkSession.builder \
        .appName("ComputeFeaturesAndLabels") \
        .config("spark.executor.memory", "64g") \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memoryOverhead", "1024") \
        .config("spark.network.timeout", "800s") \
        .config("spark.driver.maxResultSize", "64g") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()
    return spark


def load_features(base_dir):
    """加载特征数据。"""
    # Read from features directory (output of steps 06 and 07)
    msg_features_path = os.path.join(base_dir, "features", "msg_id_features_dict.pt")
    user_features_path = os.path.join(base_dir, "features", "user_features_dict.pt")
    
    if not os.path.exists(msg_features_path):
        raise FileNotFoundError(f"File not found: {msg_features_path}")
    if not os.path.exists(user_features_path):
        raise FileNotFoundError(f"File not found: {user_features_path}")
    
    msg_id_features_dict = torch.load(msg_features_path)
    user_features_dict = torch.load(user_features_path)
    print('Feature data loaded.')
    
    return msg_id_features_dict, user_features_dict


def get_user_features(user_id, user_features_dict):
    """获取用户特征。"""
    feature = user_features_dict.get(user_id)
    if feature is None:
        raise ValueError(f"User feature not found: user_id = {user_id}")
    return feature


def get_msg_features(msg_id, msg_id_features_dict):
    """获取消息特征。"""
    feature = msg_id_features_dict.get(msg_id)
    if feature is None:
        raise ValueError(f"Message feature not found: msg_id = {msg_id}")
    return feature


def process_edges(edges_df, user_features_dict, msg_id_features_dict):
    """处理边数据，提取特征。"""
    user_i_features = []
    user_j_features = []
    tweet_features = []
    
    total_records = edges_df.count()
    for row in tqdm(edges_df.collect(), total=total_records, desc="Processing edges", unit="row"):
        user_i_feature = get_user_features(row['user_i'], user_features_dict)
        user_j_feature = get_user_features(row['user_j'], user_features_dict)
        msg_feature = get_msg_features(row['tweet_id'], msg_id_features_dict)

        if isinstance(user_i_feature, torch.Tensor) and isinstance(user_j_feature, torch.Tensor) and isinstance(msg_feature, torch.Tensor):
            user_i_features.append(user_i_feature)
            user_j_features.append(user_j_feature)
            tweet_features.append(msg_feature)

    return user_i_features, user_j_features, tweet_features


def process_topic_data(spark, base_dir, topic, platform, data_type, user_features_dict, msg_id_features_dict):
    """处理单个 topic 和 data_type 的数据。"""
    print(f"\nProcessing topic: {topic}, data: {data_type}")
    
    # 清除缓存
    spark.catalog.clearCache()
    
    # Read from samples directory (output of step 04)
    positive_samples_path = os.path.join(base_dir, "samples", f"positive_samples_{data_type}")
    negative_samples_path = os.path.join(base_dir, "samples", f"negative_samples_{data_type}")
    
    if not os.path.exists(positive_samples_path):
        raise FileNotFoundError(f"File not found: {positive_samples_path}")
    if not os.path.exists(negative_samples_path):
        raise FileNotFoundError(f"File not found: {negative_samples_path}")
    
    # 加载边数据
    valid_edges_df = spark.read.csv(positive_samples_path, header=True)
    unvalid_edges_df = spark.read.csv(negative_samples_path, header=True)
    
    print('Edge data loaded.')
    print(f"valid_edges: {valid_edges_df.count()}")
    
    unvalid_edges_df = unvalid_edges_df.coalesce(10) 
    unvalid_edges_df = unvalid_edges_df.withColumnRenamed("msg_id", "tweet_id")
    print(f"unvalid_edges_all: {unvalid_edges_df.count()}")
    
    # 采样并添加标签
    unvalid_sample_df = unvalid_edges_df.sample(False, 1.0, seed=42)
    print(f"Sampled invalid edges count: {unvalid_sample_df.count()}")
    
    unvalid_sample_df = unvalid_sample_df.withColumn("y", lit(0))
    valid_edges_df = valid_edges_df.withColumn("y", lit(1))
    
    # 处理边数据
    valid_user_i_features, valid_user_j_features, valid_tweet_features = process_edges(
        valid_edges_df, user_features_dict, msg_id_features_dict
    )
    unvalid_user_i_features, unvalid_user_j_features, unvalid_tweet_features = process_edges(
        unvalid_sample_df, user_features_dict, msg_id_features_dict
    )
    
    # 合并特征
    all_user_i_features = valid_user_i_features + unvalid_user_i_features
    all_user_j_features = valid_user_j_features + unvalid_user_j_features
    all_tweet_features = valid_tweet_features + unvalid_tweet_features
    labels = [1] * len(valid_user_i_features) + [0] * len(unvalid_user_i_features)
    
    # 打印数据量
    print(f'Valid user feature count: {len(valid_user_i_features)}')
    print(f'Invalid user feature count: {len(unvalid_user_i_features)}')
    print(f'All user_i feature count: {len(all_user_i_features)}')
    print(f'All user_j feature count: {len(all_user_j_features)}')
    print(f'All message feature count: {len(all_tweet_features)}')
    
    # 保存为 .pt 文件
    batch_size = 100000
    # Save to outputs directory under base_dir
    output_dir = os.path.join(base_dir, 'outputs', f'outputs_{data_type}')
    os.makedirs(output_dir, exist_ok=True)
    
    total_records = len(all_user_i_features)
    for batch_start in range(0, total_records, batch_size):
        batch_end = min(batch_start + batch_size, total_records)
        batch_user_i_features = all_user_i_features[batch_start:batch_end]
        batch_user_j_features = all_user_j_features[batch_start:batch_end]
        batch_tweet_features = all_tweet_features[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        
        batch_file_path = os.path.join(output_dir, f'result_batch_{batch_start // batch_size}.pt')
        torch.save({
            'in_user_feature': torch.stack(batch_user_i_features),
            'out_user_feature': torch.stack(batch_user_j_features),
            'message_feature': torch.stack(batch_tweet_features),
            'y': batch_labels,
        }, batch_file_path)
        print(f"Saved batch {batch_start // batch_size} to {batch_file_path}")


def main(base_dir=None, topic='电影', platform='Twitter', data_types=None):
    """
    主函数：计算特征和标签
    
    Parameters:
    -----------
    base_dir : str, optional
        数据基础目录路径（包含 {topic}_{platform} 目录的父目录），
        如果为 None 则使用项目根目录
    topic : str, optional
        话题名称，默认为 '电影'
    platform : str, optional
        平台名称，默认为 'Twitter'
    data_types : list, optional
        数据类型列表，默认为 ['train', 'test']
    """
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    
    if data_types is None:
        data_types = ['train', 'test']
    
    topics = [topic]
    
    spark = create_spark_session()
    
    try:
        # 加载特征数据
        msg_id_features_dict, user_features_dict = load_features(base_dir)
        
        # 遍历所有topic和data组合
        for topic_item in topics:
            for data_type in data_types:
                try:
                    process_topic_data(
                        spark, base_dir, topic_item, platform, data_type,
                        user_features_dict, msg_id_features_dict
                    )
                except Exception as e:
                    print(f"Error while processing {topic_item} {data_type}: {e}")
                    continue
        
        print("\n✅ All data processed!")
        
    finally:
        try:
            spark.stop()
            print("SparkSession stopped successfully")
        except Exception as e:
            print(f"Error stopping SparkSession: {e}")
        finally:
            from pyspark import SparkContext
            SparkContext._active_spark_context = None


if __name__ == "__main__":
    main()

