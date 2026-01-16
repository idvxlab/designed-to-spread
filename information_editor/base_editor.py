# information_editor/base_editor.py
import os
import sys
import json
import re
import random
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline, DiffusionPipeline
from dotenv import load_dotenv

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Long_CLIP.model import longclip
from influence_indicator.train_pijc_model import MultiInputNet as PijcModel


class ClipFeatureExtractor(nn.Module):
    """
    Multimodal feature extractor supporting Long-CLIP based text/image encoding 
    and diffusion-based text-to-image generation. 
    """
    CONTEXT_LENGTH = 248  # Long-CLIP specific context length 
    
    def __init__(self, text_to_image_model=None, device='cuda'):
        super(ClipFeatureExtractor, self).__init__()
        self.device = torch.device(device)
        self.lock = Lock()
        
        # Load Long-CLIP model for aligned semantic space
        self.clip_model, self.preprocess = longclip.load(
            os.path.join(project_root, 'Long_CLIP', 'longclip-B.pt'),
            device=self.device
        )
        self.clip_model.eval()
        
        # Initialize generative pipeline for the Information Editor agent
        self.pipeline = self._init_pipeline(text_to_image_model)

    def _init_pipeline(self, model_type):
        if model_type is None:
            return None

        model_dir = json.loads(os.getenv("MODEL_DIR")).get(model_type)
        if not model_dir:
            raise ValueError(f"Invalid model_type: {model_type}")

        try:
            if model_type == "sdxl":
                return StableDiffusionXLPipeline.from_pretrained(
                    model_dir, torch_dtype=torch.float16, variant="fp16"
                ).to(self.device)
            elif model_type == "pixart":
                return PixArtAlphaPipeline.from_pretrained(
                    model_dir, torch_dtype=torch.float16
                ).to(self.device)
            elif model_type == "playground":
                return DiffusionPipeline.from_pretrained(
                    model_dir, torch_dtype=torch.float16, variant="fp16"
                ).to(self.device)
            # ... Add more models as needed
            
            raise ValueError(f"Unsupported model_type: {model_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_type} pipeline: {e}")

    def generate_image(self, prompt):
        """Generates image from optimized prompt using the generative agent."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        with self.lock, torch.no_grad():
            return self.pipeline(prompt=prompt, num_inference_steps=30).images[0]

    def encode_image(self, image):
        """Encodes visual content into CLIP latent space."""
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image object")
        processed_img = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.lock, torch.no_grad():
            return self.clip_model.encode_image(processed_img)

    def encode_text(self, text):
        """Encodes textual content into CLIP latent space."""
        cleaned = self.clean_text(text)
        tokens = longclip.tokenize(
            [cleaned], context_length=self.CONTEXT_LENGTH, truncate=True
        ).to(self.device)
        with self.lock, torch.no_grad():
            return self.clip_model.encode_text(tokens)

    @staticmethod
    def clean_text(text):
        """Removes social media noise (URLs, mentions, hashtags)."""
        if not isinstance(text, str): return ''
        text = re.sub(r'http\S+|www.\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[\'\"“”‘’]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return ''.join(re.findall(r"[a-zA-Z0-9\s.,!?;:()'-]", text))

class PolicyNetwork(nn.Module):
    """
    Policy network for the Information Editor to sample interpretable editing actions.
    """
    def __init__(self, action_dim, state_dim=512, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

class BaseEditor:
    """
    Core framework for Diffusion-Oriented Content Generation (DOCG).
    Integrates the Influence Indicator and Information Editor.
    """
    def __init__(self, predict_model_path, user_features_dict_path, test_users_path, output_path, text_to_image_model=None, device='cuda'):
        self.device = torch.device(device)
        self.text_to_image_model = text_to_image_model
        self.output_path = output_path
        # Influence Indicator: Pairwise Diffusion Estimator
        self.predict_model = self._load_indicator(predict_model_path)
        self.user_features, self.user_pairs = self.precompute_users(user_features_dict_path, test_users_path)

    def _load_indicator(self, model_path):
        model = PijcModel().to(self.device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    def precompute_users(self, feat_dict_path, test_users_path):
        """Precomputes target audience features for faster influence estimation."""
        user_features_dict = torch.load(feat_dict_path, weights_only=True)
        with open(test_users_path) as f:
            test_user_ids = json.load(f)
        
        random.seed(42)
        sampled_ids = random.sample(test_user_ids, min(len(test_user_ids), 200))
        
        raw_users, ui_features, uj_features = [], [], []
        for uid in sampled_ids:
            key = str(uid)
            if key in user_features_dict:
                feat = user_features_dict[key]
                raw_users.append(feat.numpy() if torch.is_tensor(feat) else feat)

        with torch.no_grad():
            users_tensor = torch.tensor(np.array(raw_users), dtype=torch.float32, device=self.device)
            # Pre-project user features for both initiator (ui) and recipient (uj) roles
            ui_features = self.predict_model.feature_extractor(
                self.predict_model.normalize(users_tensor, self.predict_model.mean[0], self.predict_model.std[0])
            ).cpu().split(1)
            uj_features = self.predict_model.feature_extractor(
                self.predict_model.normalize(users_tensor, self.predict_model.mean[1], self.predict_model.std[1])
            ).cpu().split(1)

        user_pairs = {}
        for i, user in enumerate(raw_users):
            user_key = user.tobytes()
            user_pairs[user_key] = [(ui_features[i], uj_features[j]) for j in range(len(raw_users)) if i != j]
            
        return raw_users, user_pairs

    def select_action(self, states):
        """Samples editing actions using the reparameterization strategy."""
        if not states.is_cuda and self.device.type == 'cuda':
            states = states.to(self.device)
        
        action_dist = self.policy_net(states)
        normal_samples = action_dist.rsample()
        actions = torch.tanh(normal_samples) # Bound actions within [-1, 1]
        log_probs = (action_dist.log_prob(normal_samples) - torch.log(1 - actions.pow(2) + 1e-7)).sum(dim=1)
        
        return actions.detach().cpu().numpy(), log_probs

    def calculate_influence(self, clip_features):
        """
        Influence Indicator: Computes influence score Lu(c) to measure diffusion impact.
        """
        features = clip_features.squeeze(0).cpu().numpy()[:512]
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), 'constant')

        max_avg_score = 0.0
        batch_size = 24576 

        with torch.no_grad():
            c_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            c_norm = self.predict_model.normalize(c_tensor, self.predict_model.mean[2], self.predict_model.std[2])
            c_features = self.predict_model.feature_extractor(c_norm).unsqueeze(0)

        # Iterate over initiators to find the most influential initiator
        for user_key in self.user_pairs:
            pairs = self.user_pairs[user_key]
            if not pairs: continue
            
            initiator_scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                ui_b = torch.cat([p[0] for p in batch]).to(self.device)
                uj_b = torch.cat([p[1] for p in batch]).to(self.device)
                c_b = c_features.expand(len(batch), -1)
                
                with torch.no_grad():
                    combined = torch.cat((ui_b, uj_b, c_b), dim=1)
                    # p_ij(c) prediction
                    scores = torch.softmax(self.predict_model.classifier(combined), dim=1)[:, 1]
                    initiator_scores.append(scores.cpu())
            
            if initiator_scores:
                avg_score = torch.cat(initiator_scores).mean().item()
                max_avg_score = max(max_avg_score, avg_score)

        return max_avg_score
    
    def calculate_reward(self, delta_lu, similarity):
        """
        RL Reward function: Combines influence gain and semantic fidelity.
        """
        # Eq (5) from paper: Reward edits that boost diffusion while preserving fidelity
        similarity_factor = similarity if delta_lu > 0 else (1 - similarity)
        raw_reward = delta_lu * similarity_factor
        
        # Signed magnitude for stable RL training
        signed_magnitude = torch.sqrt(torch.abs(torch.tensor(raw_reward))) * torch.sign(torch.tensor(raw_reward))
        return signed_magnitude.item()

    def update_policy(self, episode_log_probs, episode_rewards, gamma=0.99):
        """Updates policy network parameters using policy gradient methods."""
        discounted_rewards = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.stack(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, reward in zip(episode_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()