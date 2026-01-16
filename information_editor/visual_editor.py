# information_editor/visual_editor.py
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
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

from .base_editor import BaseEditor, ClipFeatureExtractor, PolicyNetwork


class VisualEditor(BaseEditor):
    """
    VisualEditor class for optimizing text-to-image prompts for social media posts using reinforcement learning and LLMs.
    Inherits from BaseEditor and implements methods for generating candidates, processing tweets,
    training a policy network, and testing optimized prompts.
    """
    def __init__(self, predict_model_path, user_features_dict_path, test_users_path, output_path, text_to_image_model='pixart', device='cuda'):
        super().__init__(predict_model_path, user_features_dict_path, test_users_path, output_path, text_to_image_model, device)
        self.feature_extractor = ClipFeatureExtractor(text_to_image_model, device)
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY is required (set env var before running).")
        self.generate_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL"), 
            api_key=os.getenv("LLM_API_KEY"),
        ) 
        self.policy_net = PolicyNetwork(action_dim=8).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.file_lock = Lock()
        self.best_globals_lock = Lock()

    def generate_initial_prompt(self, tweet_text):
        system_prompt = f"""
        You are an expert prompt engineer with a specialization in transforming tweet content into vivid, detailed text-to-image prompts. Your goal is to craft image prompts that are not only visually rich but also deeply aligned with the tweet's context and emotional tone.

        **Instructions:**
        1. **Analyze the Tweet:** Identify the core message, key elements, and emotional tone of the tweet.
        2. **Visual Conceptualization:** Translate these elements into compelling visual concepts, ensuring they are both relevant and evocative.
        3. **Craft the Prompt:** Generate a concise, 50-word image prompt that encapsulates the essence of the tweet. Include any celebrities, locations, or specific elements mentioned in the tweet.
        4. **Metaphorical Representation:** If the tweet lacks explicit visual elements, create metaphorical or symbolic visuals that align with the tweet's theme.
        5. **Tone and Style Consistency:** Ensure the visual style and tone of the prompt match the tweet's mood, whether it's celebratory, reflective, or humorous.

        **Example:**
        - **Tweet:** "Just landed my dream job! Time to celebrate! 🎉"
        - **Prompt:** "A joyful person jumping in celebration against a modern city skyline at sunset, confetti raining down, capturing the excitement of a new beginning."

        **Key Considerations:**
        - **Specificity:** Be precise in your descriptions to guide the image generation effectively.
        - **Conciseness:** Keep the prompt within 50 words while maintaining clarity and richness.
        - **Relevance:** Ensure all elements in the prompt are directly tied to the tweet's content and intent.

        **Output Format Requirements:**
        1. The output MUST be in JSON format with the following fields:
        - `tweet`: The exact original input tweet.
        - `prompt`: The final, optimized 50-word image prompt that visually represents the tweet's meaning and tone.
        2. Return ONLY the JSON object, without any additional explanations or formatting.

        By following these guidelines, you will create image prompts that not only inspire visually stunning artwork but also resonate deeply with the original tweet's message.
        """
        user_prompt = f"""
        **Tweet:**: {tweet_text}
        """
        
        max_retries = 5
        for _ in range(max_retries):
            try:
                response = self.generate_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.7,
                    max_tokens=1500,
                    response_format={'type': 'json_object'}
                )
                return json.loads(response.choices[0].message.content).get("prompt", tweet_text)
            except Exception:
                continue
        return tweet_text
    
    def generate_dynamic_prompt(self, weights):
        principles = [
            "Colorfulness: Whether the image is rich in color and has visual impact",
            "Human Scene: Whether the image includes scenes with people to enhance emotional resonance",
            "Emotion: Whether the image can evoke strong emotional responses (such as joy, shock, or being moved)",
            "Professional: Whether the image has the quality of professional photography",
            "Brightness: Whether the image is bright and attention-grabbing",
            "Clarity: Whether the image is clear and its details are prominent",
            "Visual Balance: Whether the visual elements in the image are evenly distributed",
            "Focus of the Picture: Whether the image focuses on the subject, avoiding visual dispersion",
        ]
        emphasized = [
            f"{i+1}. {principle} (Emphasis: {weight * 100:.1f}%)"
            for i, (principle, weight) in enumerate(zip(principles, weights))
        ]
        base_prompt = (
            "You are a text-to-image prompt engineer, skilled in drafting and precisely optimizing prompts to maximize the visual appeal and virality of generated images on social media platforms.\n\n"
            "Your task is to refine the provided text-to-image prompt by rewriting it in a way that fully retains the semantic content of the original tweet, while enriching background details, ensuring the image composition is vibrant, meaningful, and aesthetically pleasing.\n\n"
            "## Weight Instructions\n\n"
            "The following visual dimensions have been assigned specific weights. The weight, expressed as a percentage, indicates the intensity of optimization for the corresponding dimension, with a range from -100% to 100%:\n\n"
            "- **Positive values** indicate the need to enhance the expression in the corresponding dimension (+100% means maximum enhancement).\n"
            "- **Negative values** indicate the need to reduce the expression in the corresponding dimension (-100% means maximum reduction).\n"
            "- **0%** indicates no adjustment is needed for that dimension.\n\n"
            "## Please optimize the text-to-image prompt based on the following dimensions and their corresponding weights\n\n"
            + "\n".join(emphasized) + "\n\n"
            "## Optimization Requirements\n\n"
            "1. The optimized prompt must faithfully preserve the original tweet's semantics and intended message.\n"
            "2. Enrich the background context and visual storytelling, ensuring the scene feels alive and dynamic.\n"
            "3. Emphasize the visual vitality and aesthetic beauty of the image through natural and seamless prompt enhancements.\n"
            "4. The optimized prompt MUST NOT be identical to the original prompt; rewrite with nuanced adjustments.\n"
        )
        output_prompt = (
            "## Output Format Requirements\n\n"
            "1. The output MUST be in JSON format with the following fields:\n"
            "   - `original_tweet_text`: The exact original input tweet text.\n"
            "   - `original_prompt`: The original input text-to-image prompt before optimization.\n"
            "   - `optimized_prompt`: The refined prompt after optimization, keeping within 75 words.\n"
            "2. Return ONLY the JSON object, without any extra explanations or formatting.\n"
        )
        return base_prompt + "\n\n" + output_prompt

    def generate_candidates(self, tweet_text, prev_prompt, weights, num=1):
        system_prompt = self.generate_dynamic_prompt(weights)
        user_prompt = f"""
        Please optimize the following text-to-image prompt based on the original tweet text and given weights:
        **Original Tweet Text**: 
        {tweet_text}
        **Original Prompt**:
        {prev_prompt}
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
                    candidate = json_output.get("optimized_prompt", "")
                    
                    if candidate and candidate != prev_prompt:
                        return candidate
                    else:
                        logging.info("Generated prompt is identical to the original or empty, retrying...")
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

    def process_single_tweet(self, tweet_id, tweet_text, current_prompt, weights, orig_data, best_globals, output_path, episode, step):
        candidates = self.generate_candidates(tweet_text, current_prompt, weights)
        best_step = {
            'prompt': None,
            'image': None,
            'reward': -float('inf'),
            'lc': 0,
            'sim': 0
        }
        
        for candidate in candidates:
            cand_image = self.feature_extractor.generate_image(candidate)
            cand_features = self.feature_extractor.encode_image(cand_image)
            sim = torch.cosine_similarity(
                cand_features.to(self.device), 
                orig_data[tweet_id]['image_features'].to(self.device)
            ).item()
            lc = self.calculate_influence(cand_features)
            reward = self.calculate_reward(lc - orig_data[tweet_id]['image_lc'], sim)
            
            if reward > best_step['reward']:
                best_step.update({
                    'prompt': candidate,
                    'image': cand_image,
                    'reward': reward,
                    'lc': lc,
                    'sim': sim
                })
        
        if best_step['prompt']:
            with self.best_globals_lock:
                if best_step['reward'] > best_globals[tweet_id]['reward']:
                    best_globals[tweet_id].update({
                        'prompt': best_step['prompt'],  
                        'image': best_step['image'],
                        'reward': best_step['reward'],
                        'lc': best_step['lc'],
                        'sim': best_step['sim']
                    })
            
            log_file_path = os.path.join(output_path, str(tweet_id), "image_log.txt")
            with self.file_lock:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    print(f"Episode: {episode} | Step: {step} | Candidate: {best_step['prompt']}", 
                          file=log_file, flush=True)
                    print(f"Reward: {best_step['reward']:.4f} | Lc: {best_step['lc']:.4f} | Sim: {best_step['sim']:.4f}", 
                          file=log_file, flush=True)
        
        return tweet_id, best_step

    def train_prompt_policy(self, training_tweets, output_path, num_episodes=50, num_steps=3, 
                            start_episode=0, history=None, best_globals=None, orig_data=None, 
                            history_by_tweet=None, orig_prompts=None, orig_images=None, max_workers=200):
        os.makedirs(output_path, exist_ok=True)

        orig_data = orig_data if orig_data is not None else {} 
        best_globals = best_globals if best_globals is not None else {} 
        history_by_tweet = history_by_tweet if history_by_tweet is not None else {} 
        history = history if history is not None else {'avg_rewards': [], 'avg_lcs': [], 'avg_sims': [], 'losses': []}

        tweet_ids = list(training_tweets.keys())
        for tweet_id, tweet_text in training_tweets.items():
            if tweet_id not in orig_data:
                orig_text_features = self.feature_extractor.encode_text(tweet_text)
                orig_text_lc = self.calculate_influence(orig_text_features)
                
                if orig_prompts and tweet_id in orig_prompts:
                    orig_prompt = orig_prompts[tweet_id]
                    print(f"Using previously stored prompt for tweet {tweet_id}")
                else:
                    orig_prompt = self.generate_initial_prompt(tweet_text)
                    print(f"Generating new prompt for tweet {tweet_id}")

                if orig_images and tweet_id in orig_images:
                    orig_image = orig_images[tweet_id]
                    print(f"Using previously stored image for tweet {tweet_id}")
                else:
                    orig_image = self.feature_extractor.generate_image(orig_prompt)
                    print(f"Generating new image for tweet {tweet_id}")
                
                orig_image_features = self.feature_extractor.encode_image(orig_image)
                orig_image_lc = self.calculate_influence(orig_image_features)
                
                orig_data[tweet_id] = {
                    'text': tweet_text, 
                    'text_lc': orig_text_lc, 
                    'prompt': orig_prompt, 
                    'image': orig_image, 
                    'image_features': orig_image_features, 
                    'image_lc': orig_image_lc
                }
                best_globals[tweet_id] = {
                    'prompt': orig_prompt, 
                    'image': orig_image, 
                    'reward': 0.0, 
                    'lc': orig_image_lc, 
                    'sim': 1.0
                }
                history_by_tweet[tweet_id] = {
                    "rewards": [], 
                    "lcs": [], 
                    "sims": []
                }

                tweet_dir = os.path.join(output_path, "train", str(tweet_id))
                os.makedirs(tweet_dir, exist_ok=True)
                with open(os.path.join(tweet_dir, "image_log.txt"), 'w', encoding='utf-8') as log_file:
                    print(f"Starting optimization... Text Lc: {orig_text_lc:.4f} Image Lc: {orig_image_lc:.4f}", 
                          file=log_file, flush=True)

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
                current_prompts = {tweet_id: orig_data[tweet_id]['prompt'] for tweet_id in tweet_ids} 
                current_images = {tweet_id: orig_data[tweet_id]['image'] for tweet_id in tweet_ids}
                states = torch.stack([
                    self.feature_extractor.encode_image(current_images[tweet_id]).squeeze(0)  
                    for tweet_id in tweet_ids
                ]).to(self.device, dtype=torch.float32)
                
                episode_log_probs = []
                episode_rewards = []
                episode_lcs = []
                episode_sims = []

                for step in range(num_steps):
                    weights, log_probs = self.select_action(states)
                    episode_log_probs.append(log_probs)
                    
                    new_prompts = {}
                    new_images = {}
                    step_rewards = []
                    step_lcs = []
                    step_sims = []
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_tweet = {
                            executor.submit(
                                self.process_single_tweet,
                                tweet_id,
                                orig_data[tweet_id]['text'],
                                current_prompts[tweet_id], 
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
                            if best_step['prompt']:
                                new_prompts[tweet_id] = best_step['prompt']
                                new_images[tweet_id] = best_step['image']
                                step_rewards.append(best_step['reward'])
                                step_lcs.append(best_step['lc'])
                                step_sims.append(best_step['sim'])
                            else:
                                new_prompts[tweet_id] = orig_data[tweet_id]['prompt']
                                new_images[tweet_id] = orig_data[tweet_id]['image']
                                step_rewards.append(0.0)  
                                step_lcs.append(orig_data[tweet_id]['image_lc'])  
                                step_sims.append(1.0)  

                    current_prompts = new_prompts
                    current_images = new_images
                    states = torch.stack([
                        self.feature_extractor.encode_image(current_images[tweet_id]).squeeze(0)  
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
                print(f"Error occurred at episode {episode}: {e}")
                save_checkpoint_internal(episode - 1 if episode > start_episode else start_episode)
                raise e
            
        torch.save(self.policy_net.state_dict(), 
                   os.path.join(output_path, f'final_model_episode{num_episodes-1}.pt'))
        save_checkpoint_internal(num_episodes - 1)
        
        return orig_data, best_globals, history, history_by_tweet

    def test_prompts(self, test_prompts, output_path, num_steps=3, orig_prompts=None, orig_images=None, max_workers=200):
        os.makedirs(output_path, exist_ok=True)
        orig_data = {}         
        best_globals = {}      
        step_images = {}       

        tweet_ids = list(test_prompts.keys())
        for tweet_id, tweet_text in test_prompts.items():
            orig_text_features = self.feature_extractor.encode_text(tweet_text)
            orig_text_lc = self.calculate_influence(orig_text_features)
            
            if orig_prompts and tweet_id in orig_prompts:
                orig_prompt = orig_prompts[tweet_id]
                print(f"Using previously stored prompt for tweet {tweet_id}")
            else:
                orig_prompt = self.generate_initial_prompt(tweet_text)
                print(f"Generating new prompt for tweet {tweet_id}")

            if orig_images and tweet_id in orig_images:
                orig_image = orig_images[tweet_id]
                print(f"Using previously stored image for tweet {tweet_id}")
            else:
                orig_image = self.feature_extractor.generate_image(orig_prompt)
                print(f"Generating new image for tweet {tweet_id}")
            
            orig_image_features = self.feature_extractor.encode_image(orig_image)
            orig_image_lc = self.calculate_influence(orig_image_features)
            
            orig_data[tweet_id] = {
                'text': tweet_text, 
                'text_lc': orig_text_lc, 
                'prompt': orig_prompt, 
                'image': orig_image, 
                'image_features': orig_image_features, 
                'image_lc': orig_image_lc
            }
            best_globals[tweet_id] = {
                'prompt': None,
                'image': None,
                'reward': -float('inf'),
                'lc': 0,
                'sim': 0
            }
            step_images[tweet_id] = [] 
            
            tweet_dir = os.path.join(output_path, str(tweet_id))
            os.makedirs(tweet_dir, exist_ok=True)
            with open(os.path.join(tweet_dir, "image_log.txt"), 'w', encoding='utf-8') as log_file:
                print(f"Starting optimization... Text Lc: {orig_text_lc:.4f} Image Lc: {orig_image_lc:.4f}", 
                      file=log_file, flush=True)
            
            orig_image_dir = os.path.join(tweet_dir, "images")
            os.makedirs(orig_image_dir, exist_ok=True)
            orig_image_path = os.path.join(orig_image_dir, "original_image.png")
            orig_image.save(orig_image_path)
            print(f"Saved original image for tweet {tweet_id} at {orig_image_path}")

        episode = 0
        current_prompts = {tweet_id: orig_data[tweet_id]['prompt'] for tweet_id in tweet_ids}
        current_images = {tweet_id: orig_data[tweet_id]['image'] for tweet_id in tweet_ids}
        states = torch.stack([
            self.feature_extractor.encode_image(current_images[tweet_id]).squeeze(0)  
            for tweet_id in tweet_ids
        ]).to(self.device, dtype=torch.float32)
        
        for step in range(num_steps):
            print(f"Test Step {step + 1}/{num_steps}")
            weights, _ = self.select_action(states)
            new_prompts = {}
            new_images = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tweet = {
                    executor.submit(
                        self.process_single_tweet,
                        tweet_id,
                        orig_data[tweet_id]['text'],
                        current_prompts[tweet_id], 
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
                    if best_step['prompt']:
                        new_prompts[tweet_id] = best_step['prompt']
                        new_images[tweet_id] = best_step['image']
                        
                        if best_step['image'] is not None:
                            tweet_dir = os.path.join(output_path, str(tweet_id))
                            step_image_dir = os.path.join(tweet_dir, "images")
                            os.makedirs(step_image_dir, exist_ok=True)
                            
                            step_image_path = os.path.join(step_image_dir, f"step_{step+1}_image.png")
                            if isinstance(best_step['image'], Image.Image):
                                best_step['image'].save(step_image_path)
                            else:
                                Image.fromarray(best_step['image']).save(step_image_path)
                            
                            print(f"Saved step {step+1} image for tweet {tweet_id} at {step_image_path}")
                            step_images[tweet_id].append({
                                'step': step + 1,
                                'image_path': step_image_path,
                                'prompt': best_step['prompt'],
                                'reward': best_step['reward'],
                                'lc': best_step['lc'],
                                'sim': best_step['sim']
                            })
                    else:
                        new_prompts[tweet_id] = current_prompts[tweet_id]
                        new_images[tweet_id] = current_images[tweet_id]

            current_prompts = new_prompts
            current_images = new_images
            states = torch.stack([
                self.feature_extractor.encode_image(current_images[tweet_id]).squeeze(0)  
                for tweet_id in tweet_ids
            ]).to(self.device, dtype=torch.float32)

        for tweet_id in tweet_ids:
            if best_globals[tweet_id]['image'] is not None:
                tweet_dir = os.path.join(output_path, str(tweet_id))
                best_image_dir = os.path.join(tweet_dir, "images")
                os.makedirs(best_image_dir, exist_ok=True)
                
                best_image_path = os.path.join(best_image_dir, "best_image.png")
                if isinstance(best_globals[tweet_id]['image'], Image.Image):
                    best_globals[tweet_id]['image'].save(best_image_path)
                else:
                    Image.fromarray(best_globals[tweet_id]['image']).save(best_image_path)
                
                print(f"Saved best image for tweet {tweet_id} at {best_image_path}")

        return orig_data, best_globals