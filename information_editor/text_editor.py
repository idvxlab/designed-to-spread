# information_editor/text_editor.py
import concurrent.futures
import glob
import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import torch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from .base_editor import BaseEditor, ClipFeatureExtractor, PolicyNetwork


class TextEditor(BaseEditor):
    """
    TextEditor class for optimizing social media text posts using reinforcement learning and LLMs.
    Inherits from BaseEditor and implements methods for generating candidates, processing tweets,
    training a policy network, and testing optimized texts.
    """
    def __init__(self, predict_model_path, user_features_dict_path, test_users_path, output_path, text_to_image_model=None, device='cuda'):
        super().__init__(predict_model_path, user_features_dict_path, test_users_path, output_path, text_to_image_model, device)
        self.feature_extractor = ClipFeatureExtractor(text_to_image_model, device)
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY is required (set env var before running).")
        self.generate_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
        ) 
        self.policy_net = PolicyNetwork(action_dim=6).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.file_lock = Lock()
        self.best_globals_lock = Lock()

    def generate_dynamic_prompt(self, weights):
        principles = [
            "Social Currency: Does the content enhance the sharer's image (e.g., look good, intelligent, funny, or like a trendsetter)? ",
            "Triggers: Is the content strongly connected to real-life scenarios, promoting frequent recall?",
            "Emotion: Does the content evoke high-arousal emotions (e.g., excitement, surprise, delight)?",
            "Public: Is the content format easily shareable and visible?",
            "Practical Value: Does the content offer concrete guidelines or useful information?",
            "Stories: Does the content include a narrative framework or memorable elements?"
        ]
        emphasized = [
            f"{i+1}. {principle} (Emphasis: {weight * 100:.1f}%)"
            for i, (principle, weight) in enumerate(zip(principles, weights))
        ]
        base_prompt = (
            "You are an expert in social media content crafting, adept at subtly fine-tuning text to amplify its shareability while keeping it natural and authentic.\n\n"
            "Your task is to optimize the following social media post, making it more likely to be shared, while fully preserving the original intent, tone, and personal voice of the message.\n\n"
            "This is a real social media post written by a human and meant to be shared by other humans. Any adjustments you make should feel seamless, as if the original author could have written them naturally.\n\n"
            "## Weight Instructions\n\n"
            "You will optimize the post according to Jonah Berger's STEPPS framework, adjusting six specific dimensions. Each dimension has an associated weight ranging from -100% to +100%, indicating how much its influence should be reduced or enhanced:\n\n"
            "- **Positive values** indicate the need to enhance the expression in the corresponding dimension (+100% means maximum enhancement).\n"
            "- **Negative values** indicate the need to reduce the expression in the corresponding dimension (-100% means maximum reduction).\n"
            "- **0%** indicates no adjustment is needed for that dimension.\n\n"
            "## Please Precisely Optimize the Tweet Based on the Following STEPPS Dimensions and Their Corresponding Weights\n\n"
            + "\n".join(emphasized) + "\n\n"
            "## Optimization Requirements\n\n"
            "1. The original meaning, tone, and personal expression style of the post must remain intact.\n"
            "2. Ensure all edits blend smoothly into the text, without sounding forced or artificial.\n"
            "3. Maintain the original language of the post (e.g., if it’s in English, the output must be in English).\n"
            "4. The optimized text must have visible adjustments, and cannot be identical to the original text.\n"
            "5. The optimized version should feature diverse sentence structures and vivid, flexible expressions, avoiding monotony or rigid patterns.\n"
            "6. Ensure the language flows smoothly with clear logic and highlighted key points, making the message more impactful.\n"
            "7. Where appropriate, you may add emojis to enhance emotional resonance and visual appeal, as long as they align with the original tone.\n"
            "8. Remember, this is a social media post that should feel authentic, engaging, and human.\n"
        )
        output_prompt = (
            "## Output Format Requirements\n\n"
            "1. The output MUST be in JSON format with the following fields:\n"
            "   - `original_text`: The exact original input text.\n"
            "   - `optimized_text`: The human-like, optimized version of the text.\n"
            "2. Return ONLY the JSON object, without any additional explanations or formatting.\n"
        )
        return base_prompt + "\n\n" + output_prompt

    def generate_candidates(self, tweet_text, weights, num=1):
        system_prompt = self.generate_dynamic_prompt(weights)
        user_prompt = f"""
        Please optimize the following tweet text based on the given STEPPS weights:
        ```{tweet_text}```
        Return the result strictly following the Output Format Requirements.
        """
        candidates = []
        max_retries = 100

        def generate_candidate(_):
            retries = 0
            while retries < max_retries:
                try:
                    response = self.generate_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500,
                        response_format={'type': 'json_object'}
                    )
                    json_output = json.loads(response.choices[0].message.content)
                    candidate = json_output.get("optimized_text", "")
                    
                    if candidate and candidate != tweet_text:
                        return candidate
                    else:
                        logging.info("Generated text is identical to the original or empty, retrying...")
                        retries += 1
                
                except Exception as e:
                    print(f"Generation error: {str(e)}, retrying...")
                    retries += 1
            return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_candidate = {executor.submit(generate_candidate, _): _ for _ in range(num)}
            for future in concurrent.futures.as_completed(future_to_candidate):
                result = future.result()
                if result:
                    candidates.append(result)

        return candidates

    def process_single_tweet(self, tweet_id, current_text, weights, orig_data, best_globals, output_path, episode, step):
        candidates = self.generate_candidates(current_text, weights)
        best_step = {
            'text': None,
            'reward': -float('inf'),
            'lc': 0,
            'sim': 0
        }
        
        for candidate in candidates:
            cand_features = self.feature_extractor.encode_text(candidate)
            sim = torch.cosine_similarity(
                cand_features.to(self.device), 
                orig_data[tweet_id]['features'].to(self.device)
            ).item()
            lc = self.calculate_influence(cand_features)
            reward = self.calculate_reward(lc - orig_data[tweet_id]['lc'], sim)
            
            if reward > best_step['reward']:
                best_step.update({
                    'text': candidate,
                    'reward': reward,
                    'lc': lc,
                    'sim': sim
                })
        
        if best_step['text']:
            with self.best_globals_lock:
                if best_step['reward'] > best_globals[tweet_id]['reward']:
                    best_globals[tweet_id].update({
                        'text': best_step['text'],
                        'reward': best_step['reward'],
                        'lc': best_step['lc'],
                        'sim': best_step['sim']
                    })
            
            log_file_path = os.path.join(output_path, str(tweet_id), "text_log.txt")
            with self.file_lock:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    print(f"Episode: {episode} | Step: {step} | Candidate: {best_step['text']}", 
                          file=log_file, flush=True)
                    print(f"Reward: {best_step['reward']:.4f} | Lc: {best_step['lc']:.4f} | Sim: {best_step['sim']:.4f}", 
                          file=log_file, flush=True)
        
        return tweet_id, best_step

    def train_text_policy(self, training_tweets, output_path, num_episodes=50, num_steps=3, 
                          start_episode=0, history=None, best_globals=None, orig_data=None, 
                          history_by_tweet=None, max_workers=200):
        os.makedirs(output_path, exist_ok=True)
        orig_data = orig_data if orig_data is not None else {} 
        best_globals = best_globals if best_globals is not None else {} 
        history_by_tweet = history_by_tweet if history_by_tweet is not None else {} 
        history = history if history is not None else {'avg_rewards': [], 'avg_lcs': [], 'avg_sims': [], 'losses': []}

        tweet_ids = list(training_tweets.keys())
        for tweet_id, tweet_text in training_tweets.items():
            if tweet_id not in orig_data:
                orig_features = self.feature_extractor.encode_text(tweet_text)
                orig_lc = self.calculate_influence(orig_features)
                orig_data[tweet_id] = {
                    'text': tweet_text, 
                    'features': orig_features, 
                    'lc': orig_lc
                }
                best_globals[tweet_id] = {
                    'text': tweet_text, 
                    'reward': 0.0, 
                    'lc': orig_lc, 
                    'sim': 1.0
                }
                history_by_tweet[tweet_id] = {
                    "rewards": [], 
                    "lcs": [], 
                    "sims": []
                }

                tweet_dir = os.path.join(output_path, "train", str(tweet_id))
                os.makedirs(tweet_dir, exist_ok=True)
                with open(os.path.join(tweet_dir, "text_log.txt"), 'w', encoding='utf-8') as log_file:
                    print(f"Starting optimization... Text Lc: {orig_lc:.4f}", file=log_file, flush=True)

        def save_checkpoint_internal(episode):
            checkpoint = {
                'orig_data': orig_data,
                'best_globals': best_globals,
                'history': history,
                'history_by_tweet': history_by_tweet,
                'episode': episode,
                'policy_net_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            checkpoint_path = os.path.join(output_path, 'training_checkpoint.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"Checkpoint saved at episode {episode + 1}")

        for episode in range(start_episode, num_episodes):
            try:
                current_texts = {tweet_id: orig_data[tweet_id]['text'] for tweet_id in tweet_ids} 
                states = torch.stack([
                    self.feature_extractor.encode_text(current_texts[tweet_id]).squeeze(0)  
                    for tweet_id in tweet_ids
                ]).to(self.device, dtype=torch.float32)
                
                episode_log_probs = []
                episode_rewards = []
                episode_lcs = []
                episode_sims = []

                for step in range(num_steps):
                    weights, log_probs = self.select_action(states)
                    episode_log_probs.append(log_probs)
                    
                    new_texts = {}
                    step_rewards = []
                    step_lcs = []
                    step_sims = []
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_tweet = {
                            executor.submit(
                                self.process_single_tweet,
                                tweet_id,
                                current_texts[tweet_id], 
                                weights[i],
                                orig_data,
                                best_globals,
                                os.path.join(output_path, "train"),
                                episode,
                                step
                            ): tweet_id for i, tweet_id in enumerate(tweet_ids)
                        }
                        
                        results = {}
                        for future in as_completed(future_to_tweet):
                            tweet_id, best_step = future.result()
                            results[tweet_id] = best_step
                        
                        for tweet_id in tweet_ids:
                            best_step = results[tweet_id]
                            if best_step['text']:
                                new_texts[tweet_id] = best_step['text']
                                step_rewards.append(best_step['reward'])
                                step_lcs.append(best_step['lc'])
                                step_sims.append(best_step['sim'])
                            else:
                                new_texts[tweet_id] = orig_data[tweet_id]['text']
                                step_rewards.append(0.0)  
                                step_lcs.append(orig_data[tweet_id]['lc'])  
                                step_sims.append(1.0)  

                    current_texts = new_texts
                    states = torch.stack([
                        self.feature_extractor.encode_text(current_texts[tweet_id]).squeeze(0)  
                        for tweet_id in tweet_ids
                    ]).to(self.device, dtype=torch.float32)
                    
                    episode_rewards.append(torch.FloatTensor(step_rewards).to(self.device))
                    episode_lcs.append(torch.FloatTensor(step_lcs).to(self.device))
                    episode_sims.append(torch.FloatTensor(step_sims).to(self.device))
                
                loss = self.update_policy(episode_log_probs, episode_rewards)
                print(f"Episode {episode + 1}, Loss: {loss}")
                history['losses'].append(loss.item() if hasattr(loss, 'item') else loss)

                avg_rewards = [torch.mean(step).item() for step in episode_rewards]  
                avg_lcs = [torch.mean(step).item() for step in episode_lcs]
                avg_sims = [torch.mean(step).item() for step in episode_sims]

                history['avg_rewards'].append(np.mean(avg_rewards))
                history['avg_lcs'].append(np.mean(avg_lcs))
                history['avg_sims'].append(np.mean(avg_sims))

                for i, tweet_id in enumerate(tweet_ids):
                    history_by_tweet[tweet_id]["rewards"].append(np.mean([step[i].item() for step in episode_rewards]))
                    history_by_tweet[tweet_id]["lcs"].append(np.mean([step[i].item() for step in episode_lcs]))  
                    history_by_tweet[tweet_id]["sims"].append(np.mean([step[i].item() for step in episode_sims])) 

                current_avg_reward = history['avg_rewards'][-1]
                if current_avg_reward == max(history['avg_rewards']):
                    for old_file in glob.glob(os.path.join(output_path, 'best_model_episode*.pt')):
                        os.remove(old_file)
                    torch.save(self.policy_net.state_dict(), 
                               os.path.join(output_path, f'best_model_episode{episode}.pt'))

                save_checkpoint_internal(episode)
                print(f"Episode {episode + 1}/{num_episodes} completed. Avg Reward: {current_avg_reward:.4f}")
        
            except Exception as e:
                print(f"Error occurred at episode {episode + 1}: {e}")
                save_checkpoint_internal(episode - 1 if episode > start_episode else start_episode)
                raise e

        torch.save(self.policy_net.state_dict(), 
                   os.path.join(output_path, f'final_model_episode{num_episodes-1}.pt'))
        save_checkpoint_internal(num_episodes - 1)
        
        return orig_data, best_globals, history, history_by_tweet

    def test_texts(self, test_texts, output_path, num_steps=3, max_workers=200):
        os.makedirs(output_path, exist_ok=True)
        orig_data = {}
        best_globals = {}

        tweet_ids = list(test_texts.keys())
        for tweet_id, tweet_text in test_texts.items():
            orig_features = self.feature_extractor.encode_text(tweet_text)
            orig_lc = self.calculate_influence(orig_features)
            orig_data[tweet_id] = {
                'text': tweet_text, 
                'features': orig_features, 
                'lc': orig_lc
            }
            best_globals[tweet_id] = {
                'text': None,
                'reward': -float('inf'),
                'lc': 0,
                'sim': 0
            }
            
            tweet_dir = os.path.join(output_path, str(tweet_id))
            os.makedirs(tweet_dir, exist_ok=True)
            with open(os.path.join(tweet_dir, "text_log.txt"), 'w', encoding='utf-8') as log_file:
                print(f"Starting optimization... Text Lc: {orig_lc:.4f}", file=log_file, flush=True)

        episode = 0
        current_texts = {tweet_id: orig_data[tweet_id]['text'] for tweet_id in tweet_ids}
        states = torch.stack([
            self.feature_extractor.encode_text(current_texts[tweet_id]).squeeze(0)  
            for tweet_id in tweet_ids
        ]).to(self.device, dtype=torch.float32)
        
        for step in range(num_steps):
            weights, _ = self.select_action(states)
            new_texts = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tweet = {
                    executor.submit(
                        self.process_single_tweet,
                        tweet_id,
                        current_texts[tweet_id], 
                        weights[i],
                        orig_data,
                        best_globals,
                        output_path,
                        episode,
                        step
                    ): tweet_id for i, tweet_id in enumerate(tweet_ids)
                }
                
                results = {}
                for future in as_completed(future_to_tweet):
                    tweet_id, best_step = future.result()
                    results[tweet_id] = best_step
                
                for tweet_id in tweet_ids:
                    best_step = results[tweet_id]
                    if best_step['text']:
                        new_texts[tweet_id] = best_step['text']
                    else:
                        new_texts[tweet_id] = current_texts[tweet_id]
        
            current_texts = new_texts
            states = torch.stack([
                self.feature_extractor.encode_text(current_texts[tweet_id]).squeeze(0)  
                for tweet_id in tweet_ids
            ]).to(self.device, dtype=torch.float32)

        return orig_data, best_globals