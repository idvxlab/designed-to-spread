# scripts/tweet2image_preprocess_and_split.py
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from information_editor.base_editor import BaseEditor, ClipFeatureExtractor


class Preprocessor(BaseEditor):
    """
    Preprocesses tweet data by calculating text and visual influence scores,
    generating visual prompts/images, and splitting datasets into train/test sets.
    1. Text Influence Calculation: Computes text influence scores (Lu) using a feature extractor.
    2. Visual Prompt Generation: Transforms tweet content into detailed text-to-image prompts via LLM.
    3. Image Synthesis & Influence Calculation: Generates images from prompts and computes visual influence scores.
    4. Dataset Splitting: Divides data into training and testing sets based on influence levels.
    5. Output: Saves processed datasets and images to specified output directory.
    """

    def __init__(self, predict_model_path, user_features_dict_path, test_users_path, output_path, text_to_image_model=None, device='cuda'):
        super().__init__(
            predict_model_path=predict_model_path,
            user_features_dict_path=user_features_dict_path,
            test_users_path=test_users_path,
            output_path=output_path,
            text_to_image_model=text_to_image_model,
            device=device
        )
        self.feature_extractor = ClipFeatureExtractor(text_to_image_model, device)

        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY not found. Please set it in your .env file.")

        self.generate_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY")
        )

    def process_text_data(self, input_csv):
        """Calculates text influence scores for each tweet in the dataset."""
        df = pd.read_csv(input_csv)
        influence_results = {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Text Influence"):
            tweet_id = row['tweet_id']
            tweet_text = row['tweet_text']

            features = self.feature_extractor.encode_text(tweet_text)
            score = self.calculate_influence(features)
            influence_results[tweet_id] = score

        df['text_lu'] = df['tweet_id'].map(influence_results)
        return df

    def process_image_data(self, df, image_output_dir):
        """Generates images from tweet prompts and calculates visual influence scores."""
        os.makedirs(image_output_dir, exist_ok=True)

        prompt_dict, influence_dict = {}, {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Visual Data"):
            tweet_id = row['tweet_id']
            tweet_text = row['tweet_text']

            # Generate prompt and save synthesized image
            prompt = self.generate_initial_prompt(tweet_text)
            prompt_dict[tweet_id] = prompt

            image = self.feature_extractor.generate_image(prompt)
            image_path = os.path.join(image_output_dir, f"{tweet_id}.png")
            image.save(image_path)

            # Calculate influence for the generated image
            features = self.feature_extractor.encode_image(image)
            influence_dict[tweet_id] = self.calculate_influence(features)

        df['image_prompt'] = df['tweet_id'].map(prompt_dict)
        df['image_lu'] = df['tweet_id'].map(influence_dict)
        return df

    def generate_initial_prompt(self, tweet_text):
        """Generates a detailed text-to-image prompt from tweet text using LLM."""
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

    def split_datasets(self, df, output_dir, test_size=40):
        """Splits the dataset into training and testing sets based on text influence scores."""
        os.makedirs(output_dir, exist_ok=True)

        text_sorted = df.sort_values('text_lu')
        test_ids = text_sorted.head(test_size)['tweet_id'].unique()

        text_test = df[df['tweet_id'].isin(test_ids)].copy()
        text_train = df[~df['tweet_id'].isin(test_ids)].copy()

        text_train.to_csv(os.path.join(output_dir, 'train_tweets.csv'), index=False)
        text_test.to_csv(os.path.join(output_dir, 'test_tweets.csv'), index=False)
        return {'train': text_train, 'test': text_test}

    def process(self, input_csv, output_dir):
        """Main processing function to handle text and image data processing and dataset splitting."""
        os.makedirs(output_dir, exist_ok=True)
        img_dir = os.path.join(output_dir, 'orig_images')

        # 1. Process Text Influence
        df_text = self.process_text_data(input_csv)

        # 2. Process Visual Generation & Influence
        df_all = self.process_image_data(df_text, img_dir)

        # 3. Final Dataset Split
        self.split_datasets(df_all, output_dir)
        print(f"Preprocessing complete. Results in: {output_dir}")
        return df_all


if __name__ == "__main__":
    # 1. Define paths
    split_dir = os.path.join(project_root, 'Data', 'tweet2image')
    predict_model = os.path.join(project_root, 'Data', 'outputs', 'model', 'model_epoch_100.pth')
    user_features = os.path.join(project_root, 'Data', 'features', 'user_features_dict.pt')
    test_users = os.path.join(project_root, 'Data', 'text_split', 'test_users.json')
    # Randomly select 200 tweets from the messages of the test dataset of influence indicator
    input_csv = os.path.join(project_root, 'Data', 'tweet2image', '200_Olympics_tweets.csv')

    # 2. Initialize Preprocessor
    preprocessor = Preprocessor(
        predict_model_path=predict_model,
        user_features_dict_path=user_features,
        test_users_path=test_users,
        output_path=split_dir,
        text_to_image_model="pixart",
        device="cuda"
    )

    # 3. Run preprocessing and splitting
    preprocessor.process(
        input_csv=input_csv,
        output_dir=split_dir
    )