import os
import sys
import pandas as pd
import re
import torch
import tqdm
import gc
import shutil

# Add Long-CLIP directory to path to import model module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Data_preprocess -> AAAI_code
longclip_dir = os.path.join(parent_dir, "Long-CLIP")
if longclip_dir not in sys.path:
    sys.path.insert(0, longclip_dir)

from model import longclip


def create_model(device, model_path):
    """
    Load Long-CLIP model
    
    Parameters:
    -----------
    device : str
        Device type ('cuda' or 'cpu')
    model_path : str
        Model file path
        
    Returns:
    --------
    tuple
        (model, preprocess)
    """
    model, preprocess = longclip.load(model_path, device=device)
    model.eval()
    return model, preprocess


def clean_text(message):
    """
    Basic text cleaning
    
    Parameters:
    -----------
    message : str
        Original text
        
    Returns:
    --------
    str
        Cleaned text
    """
    if not isinstance(message, str):
        return ''
    message = re.sub(r'http\S+|www.\S+', ' ', message)
    message = re.sub(r'@\w+', ' ', message)
    message = re.sub(r'#\w+', ' ', message)
    message = re.sub(r'[\'\"""'']', ' ', message)
    message = re.sub(r'\s+', ' ', message).strip()
    english_text = ''.join(re.findall(r"[a-zA-Z0-9\s.,!?;:()'-]", message))
    return english_text if english_text else ' '


def get_free_space_gb(path="/"):
    """Return remaining space (GB) of partition containing specified path"""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def ensure_space(path, min_free_gb=2):
    """
    Check disk space, raise exception if insufficient
    
    Parameters:
    -----------
    path : str
        Path to check
    min_free_gb : float
        Minimum free disk space threshold (GB)
    """
    free_gb = get_free_space_gb(path)
    if free_gb < min_free_gb:
        raise OSError(f"Insufficient disk space, only {free_gb:.2f} GB; please free space or change output directory.")
    tqdm.tqdm.write(f" Current remaining disk space: {free_gb:.2f} GB")


def extract_clip_features(
    input_path,
    model_path,
    output_path,
    device=None,
    micro_batch_size=512,
    dim=512,
    context_length=248,
    chunksize=10000,
    min_free_gb=2
):
    """
    Extract CLIP features from text data and save as dictionary
    
    Parameters:
    -----------
    input_path : str
        Input CSV file path (contains msg_id and text columns)
    model_path : str
        Long-CLIP model path
    output_path : str
        Output .pt file path
    device : str, optional
        Device type, auto-detected by default
    micro_batch_size : int
        Number of items sent to model each time, default 512
    dim : int
        Feature dimension, default 512
    context_length : int
        Token length limit, default 248
    chunksize : int
        CSV chunk size, default 10000
    min_free_gb : float
        Minimum free disk space threshold (GB), default 5
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f" Starting to extract CLIP features (Long-CLIP)")
    print(f" Input file: {input_path}")
    print(f" Output file: {output_path}")
    print(f"  Using device: {device}")
    
    # Check disk space
    ensure_space(os.path.dirname(output_path), min_free_gb)
    
    # Loading model
    print(" Loading model...")
    model, preprocess = create_model(device, model_path)
    
    # Initialize feature dictionary
    msg_id_features_dict = {}
    
    # First calculate total rows to determine total chunk count
    print(" Calculating total file rows...")
    total_rows = 0
    temp_reader = pd.read_csv(
        input_path,
        chunksize=chunksize,
        quotechar='"',
        on_bad_lines='skip',
        encoding='utf-8'
    )
    for chunk in tqdm.tqdm(temp_reader, desc=" Counting rows", unit="chunk"):
        total_rows += len(chunk)
    total_chunks = (total_rows + chunksize - 1) // chunksize if total_rows > 0 else 0
    print(f" Total file rows: {total_rows:,}, Estimated total chunk count: {total_chunks}")
    print()
    
    # Read CSV file (chunk processing)
    reader = pd.read_csv(
        input_path,
        chunksize=chunksize,
        quotechar='"',
        on_bad_lines='skip',
        encoding='utf-8'
    )
    
    total_processed = 0
    total_valid = 0
    
    for chunk_id, chunk in enumerate(tqdm.tqdm(reader, desc=" Processing data chunks", total=total_chunks, unit="chunk")):
        if 'text' not in chunk.columns or 'msg_id' not in chunk.columns:
            print(f"Warning: Data chunk {chunk_id} Missing text/msg_id columns, skipping")
            continue
        
        # Clean text
        chunk['text_clean'] = chunk['text'].apply(clean_text)
        # Keep all messages, even if empty string after cleaning, participate in feature extraction
        valid_chunk = chunk.copy()
        
        if valid_chunk.empty:
            print(f"Warning: Data chunk {chunk_id} is empty, skipping")
            total_processed += len(chunk)
            continue
        
        texts = valid_chunk['text_clean'].tolist()
        msg_ids = valid_chunk['msg_id'].astype(str).tolist()
        
        # Replace empty string with single space to ensure encoding
        texts = [text if text.strip() else ' ' for text in texts]
        
        total_processed += len(chunk)
        total_valid += len(valid_chunk)
        
        # Extract features
        features = torch.zeros(len(texts), dim)
        
        with torch.no_grad():
            for i in tqdm.trange(
                0, len(texts), micro_batch_size,
                desc=f" Encoding data chunk {chunk_id}",
                leave=False
            ):
                batch_texts = texts[i:i + micro_batch_size]
                tokenized = longclip.tokenize(
                    batch_texts, context_length=context_length, truncate=True
                ).to(device)
                batch_features = model.encode_text(tokenized).cpu()
                features[i:i + len(batch_texts)] = batch_features
        
        # Add features to dictionary
        for msg_id, feature in zip(msg_ids, features):
            msg_id_features_dict[msg_id] = feature
        
        # Clean memory
        del features, chunk, valid_chunk
        gc.collect()
        
        # Periodically check disk space
        if (chunk_id + 1) % 10 == 0:
            ensure_space(os.path.dirname(output_path), min_free_gb)
    
    # Save feature dictionary
    print(f" Save feature dictionary to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(msg_id_features_dict, output_path)
    
    print(f" Completed！")
    print(f"   Total processed rows: {total_processed}")
    print(f"   Valid rows: {total_valid}")
    print(f"   Feature dictionary size: {len(msg_id_features_dict)}")
    print(f"   Feature dimension: {dim}")
    print(f"   Saved to: {output_path}")


def read_spark_csv_files(input_dir):
    """
    Read multiple CSV part files from Spark output
    
    Parameters:
    -----------
    input_dir : str
        Spark output directory path
        
    Returns:
    --------
    str
        Merged temporary file path, if only one file return original file path
    """
    import glob
    
    # Find all part files (excluding _SUCCESS etc.)
    csv_files = sorted([
        f for f in glob.glob(os.path.join(input_dir, "*.csv"))
        if "part-" in os.path.basename(f)
    ])
    
    if not csv_files:
        raise FileNotFoundError(f"Input directory not found or empty: {input_dir}")
    
    if len(csv_files) == 1:
        return csv_files[0]
    
    # Merge multiple CSV files
    print(f"📚 Found {len(csv_files)} CSV part files, merging...")
    all_dfs = []
    for csv_file in tqdm.tqdm(csv_files, desc="Reading CSV files"):
        try:
            df = pd.read_csv(
                csv_file,
                quotechar='"',
                on_bad_lines='skip',
                encoding='utf-8',
                header=0  # use first row as header
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Reading file {csv_file} error: {e}, skipping")
    
    if not all_dfs:
        raise ValueError("Failed to read any CSV files")
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save as temporary file
    temp_file = os.path.join(os.path.dirname(input_dir), "temp_merged_texts.csv")
    merged_df.to_csv(temp_file, index=False, quotechar='"', encoding='utf-8')
    print(f" Merged to temporary file: {temp_file} (total {len(merged_df)} rows)")
    
    return temp_file


def main(base_dir=None):
    """Main function"""
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    
    # Input: read from output of 05_filter_texts.py
    input_dir = os.path.join(base_dir, "features", "filtered_texts_data")
    
    # Read CSV files from Spark output
    input_file = read_spark_csv_files(input_dir)
    is_temp_file = "temp_merged_texts.csv" in input_file
    
    # Model path (prefer environment variable, otherwise use user-specified absolute path)
    # Default to the specified Long-CLIP path
    default_model_path = "/Users/qianziqing/Desktop/AAAI_code/Long-CLIP/longclip-B.pt"
    model_path = os.environ.get("LONGCLIP_MODEL_PATH", default_model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found LONGCLIP_MODEL_PATH={model_path}"
        )
    
    # Output path
    output_path = os.path.join(base_dir, "features", "msg_id_features_dict.pt")
    
    try:
        # Extract features
        extract_clip_features(
            input_path=input_file,
            model_path=model_path,
            output_path=output_path,
            micro_batch_size=512,
            dim=512,
            context_length=248
        )
    finally:
        # Clean temporary files
        if is_temp_file and os.path.exists(input_file):
            os.remove(input_file)
            print(f"  Deleted temporary file: {input_file}")


if __name__ == "__main__":
    main()
