# baselines/llm_baseline.py
import csv
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from information_editor.text_editor import TextEditor
from information_editor.visual_editor import VisualEditor


TEXT_RESULT_FIELDS = ['tweet_id', 'original_lc', 'optimized_lc', 'best_sim', 'optimized_text', 'tweet_text']
IMAGE_RESULT_FIELDS = [
    'tweet_id', 'original_text_lc', 'original_image_lc', 'optimized_lc', 
    'best_sim', 'original_prompt', 'optimized_prompt', 'tweet_text'
]

text_csv_lock = threading.Lock()
image_csv_lock = threading.Lock()

def get_existing_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    
    existing_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ids.add(row['tweet_id'])
    return existing_ids

def save_result(args, data, csv_name, fields, lock, result_type):
    csv_path = os.path.join(args.output_path, csv_name)
    
    with lock:
        existing_ids = get_existing_ids(csv_path)
        if data['tweet_id'] in existing_ids:
            print(f"{result_type} result for tweet {data['tweet_id']} exists, skipping...")
            return False

        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True

def save_text_result(args, data):
    return save_result(args, data, "text_results.csv", TEXT_RESULT_FIELDS, text_csv_lock, "Text")

def save_image_result(args, data):
    return save_result(args, data, "image_results.csv", IMAGE_RESULT_FIELDS, image_csv_lock, "Image")

def llm_optimize_text(text_editor, original_text):
    system_prompt = (
        "You are an expert in social media content crafting, adept at subtly fine-tuning text to amplify its shareability while keeping it natural and authentic.\n\n"
        "Your task is to optimize the following social media post, making it more likely to be shared, while fully preserving the original intent, tone, and personal voice of the message.\n\n"
        "This is a real social media post written by a human and meant to be shared by other humans. Any adjustments you make should feel seamless, as if the original author could have written them naturally.\n\n"
        "## Optimization Requirements\n\n"
        "1. The original meaning, tone, and personal expression style of the post must remain intact.\n"
        "2. Ensure all edits blend smoothly into the text, without sounding forced or artificial.\n"
        "3. Maintain the original language of the post (e.g., if it’s in English, the output must be in English).\n"
        "4. The optimized text must have visible adjustments, and cannot be identical to the original text.\n"
        "5. The optimized version should feature diverse sentence structures and vivid, flexible expressions, avoiding monotony or rigid patterns.\n"
        "6. Ensure the language flows smoothly with clear logic and highlighted key points, making the message more impactful.\n"
        "7. Where appropriate, you may add emojis to enhance emotional resonance and visual appeal, as long as they align with the original tone.\n"
        "8. Remember, this is a social media post that should feel authentic, engaging, and human.\n"
        "## Output Format Requirements\n\n"
        "1. The output MUST be in JSON format with the following fields:\n"
        "   - `original_text`: The exact original input text.\n"
        "   - `optimized_text`: The human-like, optimized version of the text.\n"
        "2. Return ONLY the JSON object, without any additional explanations or formatting.\n"
    )

    user_prompt = f"""
    Please optimize the following tweet text for a fitness-focused audience, enhancing its diffusion effectiveness:
    ```{original_text}```
    Return the result strictly following the Output Format Requirements.
    """

    for _ in range(5):
        try:
            response = text_editor.generate_client.chat.completions.create(
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
            candidate = json_output.get("optimized_text", "").strip()

            if candidate and candidate != original_text:
                orig_features = text_editor.feature_extractor.encode_text(original_text)
                orig_lc = text_editor.calculate_influence(orig_features)

                opt_features = text_editor.feature_extractor.encode_text(candidate)
                sim = torch.cosine_similarity(opt_features, orig_features).item()
                opt_lc = text_editor.calculate_influence(opt_features)

                return candidate, orig_lc, opt_lc, sim
        except Exception as e:
            print(f"Generation error: {str(e)}, retrying...")
    return None, None, None, None

def llm_optimize_prompt(visual_editor, original_text, initial_prompt, initial_image=None):
    system_prompt = (
        "You are a text-to-image prompt engineer, skilled in drafting and precisely optimizing prompts to maximize the visual appeal and virality of generated images on social media platforms.\n\n"
        "Your task is to refine the provided text-to-image prompt by rewriting it in a way that fully retains the semantic content of the original tweet, while enriching background details, ensuring the image composition is vibrant, meaningful, and aesthetically pleasing.\n\n"
        "## Optimization Requirements\n\n"
        "1. The optimized prompt must faithfully preserve the original tweet's semantics and intended message.\n"
        "2. Enrich the background context and visual storytelling, ensuring the scene feels alive and dynamic.\n"
        "3. Emphasize the visual vitality and aesthetic beauty of the image through natural and seamless prompt enhancements.\n"
        "4. The optimized prompt MUST NOT be identical to the original prompt; rewrite with nuanced adjustments.\n"
        "## Output Format Requirements\n\n"
        "1. The output MUST be in JSON format with the following fields:\n"
        "   - `original_tweet_text`: The exact original input tweet text.\n"
        "   - `original_prompt`: The original input text-to-image prompt before optimization.\n"
        "   - `optimized_prompt`: The refined prompt after optimization, keeping within 75 words.\n"
        "2. Return ONLY the JSON object, without any extra explanations or formatting.\n"
    )

    user_prompt = f"""
    Please optimize the following text-to-image prompt based on the original tweet text for a fitness-focused audience, enhancing its diffusion effectiveness:
    **Original Tweet Text**: 
    {original_text}
    **Original Prompt**:
    {initial_prompt}
    Return the result strictly following the Output Format Requirements.
    """

    for _ in range(5):
        try:
            response = visual_editor.generate_client.chat.completions.create(
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
            candidate = json_output.get("optimized_prompt", "").strip()

            if candidate and candidate != initial_prompt:
                orig_text_features = visual_editor.feature_extractor.encode_text(original_text)
                orig_text_lc = visual_editor.calculate_influence(orig_text_features)

                orig_image = initial_image if initial_image else visual_editor.feature_extractor.generate_image(initial_prompt)
                orig_image_features = visual_editor.feature_extractor.encode_image(orig_image)
                orig_image_lc = visual_editor.calculate_influence(orig_image_features)

                opt_image = visual_editor.feature_extractor.generate_image(candidate)
                opt_features = visual_editor.feature_extractor.encode_image(opt_image)
                sim = torch.cosine_similarity(opt_features, orig_image_features).item()
                opt_lc = visual_editor.calculate_influence(opt_features)

                return (candidate, orig_text_lc, orig_image_lc, opt_lc, sim, orig_image, opt_image)
        except Exception as e:
            print(f"Generation error: {str(e)}, retrying...")
    return [None] * 7

def process_single_tweet(args, tweet_id, tweet_text, default_csv, orig_img_folder):
    tweet_output_dir = os.path.join(args.output_path, str(tweet_id))
    os.makedirs(tweet_output_dir, exist_ok=True)

    def run_text_optimization():
        if tweet_id in get_existing_ids(os.path.join(args.output_path, "text_results.csv")):
            return
        
        text_editor = TextEditor(
            predict_model_path=args.predict_model,
            user_features_dict_path=args.user_features_dict_path,
            test_users_path=args.test_users_path,
            output_path=args.output_path,
            device=args.device
        )
        opt_text, lc_orig, lc_opt, sim = llm_optimize_text(text_editor, tweet_text)
        if opt_text:
            save_text_result(args, {
                'tweet_id': tweet_id, 'original_lc': lc_orig, 'optimized_lc': lc_opt,
                'best_sim': sim, 'optimized_text': opt_text, 'tweet_text': tweet_text
            })

    def run_image_optimization():
        if tweet_id in get_existing_ids(os.path.join(args.output_path, "image_results.csv")):
            return

        visual_editor = VisualEditor(
            predict_model_path=args.predict_model,
            user_features_dict_path=args.user_features_dict_path,
            test_users_path=args.test_users_path,
            output_path=args.output_path,
            device=args.device
        )

        initial_prompt = None
        if os.path.exists(default_csv):
            with open(default_csv, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row['tweet_id'] == str(tweet_id):
                        initial_prompt = row.get('image_prompt')
                        break

        initial_prompt = initial_prompt or visual_editor.generate_initial_prompt(tweet_text)
        img_path = os.path.join(orig_img_folder, f"{tweet_id}.png")
        initial_image = Image.open(img_path) if os.path.exists(img_path) else None

        res = llm_optimize_prompt(visual_editor, tweet_text, initial_prompt, initial_image)
        opt_prompt, lc_t_orig, lc_i_orig, lc_opt, sim, img_orig, img_opt = res

        if opt_prompt:
            image_dir = os.path.join(tweet_output_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            img_orig.save(os.path.join(image_dir, "orig_image.png"))
            img_opt.save(os.path.join(image_dir, "best_image.png"))

            save_image_result(args, {
                'tweet_id': tweet_id, 'original_text_lc': lc_t_orig, 'original_image_lc': lc_i_orig,
                'optimized_lc': lc_opt, 'best_sim': sim, 'original_prompt': initial_prompt,
                'optimized_prompt': opt_prompt, 'tweet_text': tweet_text
            })

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if args.text: futures.append(executor.submit(run_text_optimization))
        if args.image: futures.append(executor.submit(run_image_optimization))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in tweet {tweet_id}: {str(e)}")

def load_and_sample_tweets(csv_path, subset_size, random_seed):
    with open(csv_path, 'r', encoding='utf-8') as f:
        tweets = [(row['tweet_id'], row['tweet_text']) for row in csv.DictReader(f)]
    random.seed(random_seed)
    return random.sample(tweets, min(subset_size, len(tweets)))

if __name__ == "__main__":
    import argparse

    DEFAULT_MESSAGES_CSV = os.path.join(project_root, 'Data', 'tweet2image', 'test_tweets.csv')
    DEFAULT_PREDICT_MODEL = os.path.join(project_root, 'Data', 'outputs', 'model', 'model_epoch_100.pth')
    DEFAULT_USER_FEATURES_DICT_PATH = os.path.join(project_root, 'Data', 'features', 'user_features_dict.pt')
    DEFAULT_TEST_USERS_PATH = os.path.join(project_root, 'Data', 'text_split', 'test_users.json')
    DEFAULT_OUTPUT_PATH = os.path.join(project_root, 'results', 'baselines', 'llm_baseline')
    DEFAULT_ORIG_IMGAE_FOLDER = os.path.join(project_root, 'Data', 'tweet2image', 'orig_images')

    parser = argparse.ArgumentParser(description='Batch Tweet Optimizer')
    parser.add_argument('--messages_csv', default=DEFAULT_MESSAGES_CSV, help=f'Path to input messages CSV file')
    parser.add_argument('--subset_size', type=int, default=40, help=f'Number of tweets to sample')
    parser.add_argument('--random_seed', type=int, default=42, help=f'Random seed for sampling')
    parser.add_argument('--text', action='store_true', help='Generate optimized result as text')
    parser.add_argument('--image', action='store_true', help='Generate optimized result as image')
    parser.add_argument('--predict_model', type=str, default=DEFAULT_PREDICT_MODEL, help="Path to pretrained pijc model")
    parser.add_argument('--user_features_dict_path', type=str, default=DEFAULT_USER_FEATURES_DICT_PATH, help="Path to the precomputed user features dictionary (torch .pt file)")
    parser.add_argument('--test_users_path', type=str, default=DEFAULT_TEST_USERS_PATH, help="Path to JSON file containing test user IDs")
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save training and testing results")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training/testing (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum parallel workers')

    args = parser.parse_args()
    if not (args.text or args.image):
        args.text = True
        args.image = True

    os.makedirs(args.output_path, exist_ok=True)
    tweets = load_and_sample_tweets(args.messages_csv, args.subset_size, args.random_seed)

    print(f"Processing {len(tweets)} tweets...")
    for tweet_id, tweet_text in tweets:
        try:
            process_single_tweet(args, tweet_id, tweet_text, DEFAULT_MESSAGES_CSV, DEFAULT_ORIG_IMGAE_FOLDER)
        except Exception as e:
            print(f"Failed tweet {tweet_id}: {str(e)}")