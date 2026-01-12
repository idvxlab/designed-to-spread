#!/usr/bin/env python3
"""
Main script to run all data preprocessing steps (1-8) in sequence.

Steps:
1. extract_tables - Extract interaction, message, and user tables
2. extract_network - Extract network structure and valid edges
3. split_dataset - Split dataset into train/test sets (ensuring no overlap in text and users)
4. generate_samples - Generate positive and negative samples
5. filter_texts - Filter tweets and texts based on edge data
6. compute_message_features - Compute CLIP features for messages
7. compute_user_features - Compute user features from message features
8. compute_features_and_labels - Compute features and labels for training
"""

import os
import sys
import traceback
from datetime import datetime


def print_step_header(step_num, step_name):
    """Print formatted step header."""
    print("\n" + "=" * 80)
    print(f"Step {step_num}: {step_name}")
    print("=" * 80 + "\n")


def print_step_footer(step_num, step_name, success=True):
    """Print formatted step footer."""
    status = "SUCCESS" if success else "FAILED"
    print(f"\n{'=' * 80}")
    print(f"Step {step_num}: {step_name} - {status}")
    print("=" * 80 + "\n")


def run_step(step_num, step_name, module_name, func_name="main", **kwargs):
    """Run a preprocessing step."""
    print_step_header(step_num, step_name)
    
    try:
        # Import the module
        module = __import__(module_name, fromlist=[func_name])
        func = getattr(module, func_name)
        
        # Run the function
        if kwargs:
            func(**kwargs)
        else:
            func()
        
        print_step_footer(step_num, step_name, success=True)
        return True
        
    except Exception as e:
        print(f"\nError in {step_name}:")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print_step_footer(step_num, step_name, success=False)
        return False


def main(base_dir=None, topic='电影', platform='Twitter', data_types=None, 
         posts_all_path=None, posts_path=None, accounts_path=None):
    """
    Main function to run all preprocessing steps in sequence.
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory path (Data directory), if None use project root/Data
    topic : str, optional
        Topic name for step 8, default '电影'
    platform : str, optional
        Platform name for step 8, default 'Twitter'
    data_types : list, optional
        Data types for step 8, default ['train', 'test']
    posts_all_path : str, optional
        posts_all.jsonl file path for step 1
    posts_path : str, optional
        posts.jsonl file path for step 1
    accounts_path : str, optional
        accounts.jsonl file path for step 1
    """
    # Set base_dir to Data directory (sibling of Data_preprocess directory) using relative path
    if base_dir is None:
        # Get Data_preprocess directory, then go to parent and join with Data
        data_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(data_preprocess_dir), "Data")
    
    if data_types is None:
        data_types = ['train', 'test']
    
    print("\n" + "=" * 80)
    print("Data Preprocessing Pipeline")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Data types: {data_types}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    steps = [
        (1, "Extract Tables", "extract_tables", "main", {
            "base_dir": base_dir,
            "posts_all_path": posts_all_path,
            "posts_path": posts_path,
            "accounts_path": accounts_path
        }),
        (2, "Extract Network", "extract_network", "main", {
            "base_dir": base_dir
        }),
        (3, "Split Dataset", "split_dataset", "main", {
            "base_dir": base_dir
        }),
        (4, "Generate Samples", "generate_samples", "main", {
            "base_dir": base_dir
        }),
        (5, "Filter Texts", "filter_texts", "main", {
            "base_dir": base_dir
        }),
        (6, "Compute Message Features", "compute_message_features", "main", {
            "base_dir": base_dir
        }),
        (7, "Compute User Features", "compute_user_features", "main", {
            "base_dir": base_dir
        }),
        (8, "Compute Features and Labels", "compute_features_and_labels", "main", {
            "base_dir": base_dir,
            "topic": topic,
            "platform": platform,
            "data_types": data_types
        }),
    ]
    
    failed_steps = []
    
    for step_num, step_name, module_name, func_name, kwargs in steps:
        success = run_step(step_num, step_name, module_name, func_name, **kwargs)
        if not success:
            failed_steps.append((step_num, step_name))
            response = input(f"\nStep {step_num} failed. Continue with next step? (y/n): ")
            if response.lower() != 'y':
                print("\nPipeline stopped by user.")
                break
    
    # Print summary
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_steps:
        print(f"\nFailed steps: {len(failed_steps)}")
        for step_num, step_name in failed_steps:
            print(f"  - Step {step_num}: {step_name}")
        print("\nPipeline completed with errors.")
        sys.exit(1)
    else:
        print("\nAll steps completed successfully!")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all data preprocessing steps")
    parser.add_argument("--base_dir", type=str, default=None, 
                       help="Base directory path (Data directory)")
    parser.add_argument("--data_types", type=str, nargs='+', default=['train', 'test'],
                       help="Data types for step 8 (default: ['train', 'test'])")
    parser.add_argument("--posts_all_path", type=str, default=None,
                       help="posts_all.jsonl file path for step 1")
    parser.add_argument("--posts_path", type=str, default=None,
                       help="posts.jsonl file path for step 1")
    parser.add_argument("--accounts_path", type=str, default=None,
                       help="accounts.jsonl file path for step 1")
    
    args = parser.parse_args()
    
    main(
        base_dir=args.base_dir,
        data_types=args.data_types,
        posts_all_path=args.posts_all_path,
        posts_path=args.posts_path,
        accounts_path=args.accounts_path
    )

