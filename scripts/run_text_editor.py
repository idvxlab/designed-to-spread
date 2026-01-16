# scripts/run_text_editor.py
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from information_editor.text_editor import TextEditor


def load_tweet_data(csv_path):
    df = pd.read_csv(csv_path)
    tweets = dict(zip(df['tweet_id'], df['tweet_text']))
    return tweets

def save_checkpoint(orig_data, best_globals, history, history_by_tweet, episode, policy_net, optimizer, output_path):
    checkpoint = {
        'orig_data': orig_data,
        'best_globals': best_globals,
        'history': history,
        'history_by_tweet': history_by_tweet,
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    checkpoint_path = os.path.join(output_path, 'training_checkpoint.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at episode {episode}")

def load_checkpoint(output_path):
    checkpoint_path = os.path.join(output_path, 'training_checkpoint.pkl')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint loaded from episode {checkpoint['episode']}")
        return checkpoint
    return None

def save_training_outputs(orig_data, best_globals, history, history_by_tweet, output_path):
    records = []
    for tweet_id in orig_data:
        record = {
            "tweet_id": tweet_id,
            "original_lc": orig_data[tweet_id]["lc"],
            "optimized_lc": best_globals[tweet_id]["lc"],
            "best_reward": best_globals[tweet_id]["reward"],
            "best_sim": best_globals[tweet_id]["sim"],
            "optimized_text": best_globals[tweet_id]["text"],
            "tweet_text": orig_data[tweet_id]["text"],
            "reward_history": history_by_tweet[tweet_id]["rewards"],
            "lc_history": history_by_tweet[tweet_id]["lcs"],
            "sim_history": history_by_tweet[tweet_id]["sims"],
            # "loss_history": history["losses"]
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_path, "train_result.csv"), index=False)

def save_test_outputs(test_results, output_path):
    df = pd.DataFrame(test_results)
    df.to_csv(os.path.join(output_path, "test_result.csv"), index=False)

def run_training(args):
    print("Loading training data...")
    training_tweets = load_tweet_data(args.train_csv)

    print("Initializing optimizer...")
    optimizer = TextEditor(
        predict_model_path=args.predict_model,
        user_features_dict_path=args.user_features_dict_path,
        test_users_path = args.test_users_path,
        output_path=args.output_path,
        device=args.device
    )

    start_episode = 0
    orig_data = None
    best_globals = None
    history = None
    history_by_tweet = None
    
    if args.resume:
        checkpoint = load_checkpoint(args.output_path)
        if checkpoint:
            orig_data = checkpoint['orig_data']
            best_globals = checkpoint['best_globals']
            history = checkpoint['history']
            history_by_tweet = checkpoint['history_by_tweet']
            start_episode = checkpoint['episode'] + 1
            
            optimizer.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            print("Policy network state restored from checkpoint")
            
            optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state restored from checkpoint")
            
            print(f"Resumed from episode {start_episode}")
        else:
            print("No checkpoint found, starting training from beginning")

    print("Training...")
    orig_data, best_globals, history, history_by_tweet = optimizer.train_text_policy(
        training_tweets, os.path.join(args.output_path), num_episodes=args.num_episodes, num_steps=args.num_steps,
        start_episode=start_episode, history=history, best_globals=best_globals, 
        orig_data=orig_data, history_by_tweet=history_by_tweet
    )

    print("Saving training outputs...")
    save_training_outputs(orig_data, best_globals, history, history_by_tweet, args.output_path)

    # Visualize history metrics: rewards, lcs, sims, losses
    window_size = max(1, min(20, len(history['avg_rewards']) // 20))
    print(f"Plotting with dynamic window_size: {window_size}")

    window = np.ones(window_size) / window_size
    episodes = list(range(len(history['avg_rewards'])))
    fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 

    avg_rewards_smooth = np.convolve(history['avg_rewards'], window, mode='valid')
    axs[0].plot(episodes, history['avg_rewards'], label="Average Reward", color='blue', alpha=0.3)
    axs[0].plot(episodes[window_size-1:], avg_rewards_smooth, label=f"Moving Avg ({window_size})", color='blue')
    axs[0].set_ylabel("Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_title("Episode-wise Average Reward")
    axs[0].grid(True)
    axs[0].legend()

    avg_lcs_smooth = np.convolve(history['avg_lcs'], window, mode='valid')
    axs[1].plot(episodes, history['avg_lcs'], label="Average Lc", color='green', alpha=0.3)
    axs[1].plot(episodes[window_size-1:], avg_lcs_smooth, label=f"Moving Avg ({window_size})", color='green')
    axs[1].set_ylabel("Lc")
    axs[1].set_xlabel("Episode")
    axs[1].set_title("Episode-wise Average Lc")
    axs[1].grid(True)
    axs[1].legend()

    avg_sims_smooth = np.convolve(history['avg_sims'], window, mode='valid')
    axs[2].plot(episodes, history['avg_sims'], label="Average Similarity", color='orange', alpha=0.3)
    axs[2].plot(episodes[window_size-1:], avg_sims_smooth, label=f"Moving Avg ({window_size})", color='orange')
    axs[2].set_ylabel("Similarity")
    axs[2].set_xlabel("Episode")
    axs[2].set_title("Episode-wise Average Similarity")
    axs[2].grid(True)
    axs[2].legend()

    losses_smooth = np.convolve(history['losses'], window, mode='valid')
    axs[3].plot(episodes, history['losses'], label="Loss", color='red', alpha=0.3)  
    axs[3].plot(episodes[window_size-1:], losses_smooth, label=f"Moving Avg ({window_size})", color='red')
    axs[3].set_ylabel("Loss")
    axs[3].set_xlabel("Episode")
    axs[3].set_title("Episode-wise Training Loss")
    axs[3].grid(True)
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, "training_metrics.png"))
    plt.show()

    return optimizer  # Return the trained object

def run_testing(args, trained_optimizer):
    print("Loading test data...")
    test_texts = load_tweet_data(args.test_csv)

    # Use the test_texts function to get results
    orig_data, best_results = trained_optimizer.test_texts(test_texts, os.path.join(args.output_path, "test"), num_steps=args.num_steps)

    # Prepare test results for saving
    test_results = []
    for tweet_id in test_texts.keys():
        test_results.append({
            "tweet_id": tweet_id,
            "original_lc": orig_data[tweet_id]['lc'],
            "optimized_lc": best_results[tweet_id]["lc"],
            "best_reward": best_results[tweet_id]["reward"],
            "best_sim": best_results[tweet_id]["sim"],
            "optimized_text": best_results[tweet_id]["text"],
            "tweet_text": test_texts[tweet_id],
        })

    print("Saving test results...")
    save_test_outputs(test_results, args.output_path)

if __name__ == "__main__":
    import argparse

    DEFAULT_TRAIN_CSV = os.path.join(project_root, 'Data', 'tweet2image', 'train_tweets.csv')
    DEFAULT_TEST_CSV = os.path.join(project_root, 'Data', 'tweet2image', 'test_tweets.csv')
    DEFAULT_PREDICT_MODEL = os.path.join(project_root, 'Data', 'outputs', 'model', 'model_epoch_100.pth')
    DEFAULT_USER_FEATURES_DICT_PATH = os.path.join(project_root, 'Data', 'features', 'user_features_dict.pt')
    DEFAULT_TEST_USERS_PATH = os.path.join(project_root, 'Data', 'text_split', 'test_users.json')
    DEFAULT_OUTPUT_PATH = os.path.join(project_root, 'results', 'text_editor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=DEFAULT_TRAIN_CSV, help="Path to training CSV (must have tweet_id, tweet_text columns)")
    parser.add_argument('--test_csv', type=str, default=DEFAULT_TEST_CSV, help="Path to test CSV (must have tweet_id, tweet_text columns)")
    parser.add_argument('--predict_model', type=str, default=DEFAULT_PREDICT_MODEL, help="Path to pretrained pijc model")
    parser.add_argument('--user_features_dict_path', type=str, default=DEFAULT_USER_FEATURES_DICT_PATH, help="Path to the precomputed user features dictionary (torch .pt file)")
    parser.add_argument('--test_users_path', type=str, default=DEFAULT_TEST_USERS_PATH, help="Path to JSON file containing test user IDs")
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save training and testing results")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training/testing (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--num_episodes', type=int, default=350, help="Total number of training episodes")
    parser.add_argument('--num_steps', type=int, default=3, help="Total number of steps per episode")
    parser.add_argument('--resume', action='store_true', help="Resume training from last checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    trained_optimizer = run_training(args)

    print("Reloading trained policy model for testing...")
    test_optimizer = TextEditor(
        predict_model_path=args.predict_model,
        user_features_dict_path=args.user_features_dict_path,
        test_users_path = args.test_users_path,
        output_path=args.output_path,
        device=args.device
    )

    latest_model_path = os.path.join(args.output_path, f"final_model_episode{args.num_episodes-1}.pt")
    test_optimizer.policy_net.load_state_dict(torch.load(latest_model_path, map_location=args.device, weights_only=True))
    print(f"Loaded latest model: {latest_model_path}")
    test_optimizer.policy_net.eval()

    run_testing(args, test_optimizer)