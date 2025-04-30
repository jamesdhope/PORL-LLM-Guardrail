import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Dict
import gymnasium as gym
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

@dataclass
class Trajectory:
    states: List[str]  # Sequence of states (prompts + responses)
    actions: List[str]  # Sequence of generated responses
    rewards: List[float]  # Sequence of rewards
    log_probs: List[torch.Tensor]  # Sequence of log probabilities

class TopicEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Define target topics and their descriptions
        self.target_topics = {
            "reinforcement_learning": [
                "reinforcement learning",
                "RL algorithms",
                "policy gradient",
                "Q-learning",
                "reward function",
                "environment interaction",
                "agent behavior",
                "state-action pairs",
                "exploration vs exploitation",
                "value function"
            ],
            "ai_ethics": [
                "AI safety",
                "ethical considerations",
                "bias in AI",
                "fairness",
                "transparency",
                "accountability",
                "privacy concerns",
                "human values",
                "AI alignment",
                "responsible AI"
            ],
            "machine_learning": [
                "supervised learning",
                "unsupervised learning",
                "neural networks",
                "deep learning",
                "model training",
                "feature engineering",
                "model evaluation",
                "overfitting",
                "generalization",
                "hyperparameter tuning"
            ]
        }
        
        # Pre-compute topic embeddings
        self.topic_embeddings = self._compute_topic_embeddings()
    
    def _compute_topic_embeddings(self) -> Dict[str, torch.Tensor]:
        """Compute and store embeddings for all topic descriptions."""
        topic_embeddings = {}
        for topic, descriptions in self.target_topics.items():
            # Get embeddings for all descriptions
            embeddings = []
            for desc in descriptions:
                inputs = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling to get sentence embedding
                    embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                    embeddings.append(embedding)
            
            # Average embeddings for the topic
            topic_embedding = torch.mean(torch.stack(embeddings), dim=0)
            topic_embeddings[topic] = topic_embedding
            
        return topic_embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a single text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return self._mean_pooling(outputs, inputs['attention_mask'])
    
    def calculate_topic_similarity(self, text: str) -> float:
        """Calculate similarity between text and target topics."""
        text_embedding = self.get_embedding(text)
        
        # Calculate similarity with each topic
        similarities = []
        for topic_embedding in self.topic_embeddings.values():
            similarity = F.cosine_similarity(text_embedding, topic_embedding, dim=1)
            similarities.append(similarity.item())
        
        # Return maximum similarity (how close to any target topic)
        return max(similarities)

class RewardModel:
    def __init__(self):
        self.topic_model = TopicEmbeddingModel()
        
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward based on topic similarity."""
        # Calculate topic similarity for the response
        topic_similarity = self.topic_model.calculate_topic_similarity(response)
        
        # Calculate coherence with prompt
        prompt_embedding = self.topic_model.get_embedding(prompt)
        response_embedding = self.topic_model.get_embedding(response)
        coherence = F.cosine_similarity(prompt_embedding, response_embedding, dim=1).item()
        
        # Combine topic similarity and coherence
        # Topic similarity is more important (0.7 weight) than coherence (0.3 weight)
        reward = 0.7 * topic_similarity + 0.3 * coherence
        
        return reward

class QwenRLAgent:
    def __init__(self, model_name: str = "Qwen/Qwen-1_8B", num_trajectories: int = 4):
        print(f"Initializing QwenRLAgent with model: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True
        ).to(self.device)
        print("Model loaded successfully")
        
        self.num_trajectories = num_trajectories
        self.reward_model = RewardModel()
        
        # Freeze the model parameters
        print("Freezing model parameters...")
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Add a small policy head for response generation
        print("Initializing policy head...")
        self.policy_head = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)
        self.policy_head.to(self.device)
        print("Policy head initialized")

    def save_model(self, path: str):
        """Save the reinforced model and policy head."""
        # Save the policy head
        torch.save(self.policy_head.state_dict(), f"{path}/policy_head.pt")
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save the model config
        self.model.config.save_pretrained(path)
        
        print(f"Model saved to {path}")

    def generate_response(self, prompt: str) -> str:
        print(f"\nGenerating response for prompt: {prompt[:50]}...")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to the correct device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        print(f"Input tensor device: {inputs['input_ids'].device}")
        
        # Generate response
        print("Generating text...")
        try:
            with torch.no_grad():
                print("Starting generation with parameters:")
                print(f"- max_new_tokens: 128")
                print(f"- temperature: 0.7")
                print(f"- top_p: 0.9")
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,  # Reduced from 512
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                print("Generation completed successfully")
                
                response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                print(f"Generated response length: {len(response)} characters")
                print(f"Generated response: {response[:100]}...")
                return response[len(prompt):]  # Return only the generated part
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            print("Falling back to CPU...")
            # Try on CPU as fallback
            self.model = self.model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt):]

    def generate_trajectories(self, prompt: str) -> List[Tuple[str, float]]:
        print(f"\nGenerating {self.num_trajectories} trajectories...")
        trajectories = []
        for i in range(self.num_trajectories):
            print(f"\nTrajectory {i+1}/{self.num_trajectories}")
            response = self.generate_response(prompt)
            print("Calculating reward...")
            reward = self.reward_model.calculate_reward(prompt, response)
            print(f"Reward: {reward:.4f}")
            trajectories.append((response, reward))
        return trajectories

    def train_step(self, prompt: str, target_response: str) -> float:
        print("\nStarting training step...")
        # Generate trajectories
        trajectories = self.generate_trajectories(prompt)
        
        # Calculate PPO loss
        print("\nCalculating PPO loss...")
        loss = 0.0
        for i, (response, reward) in enumerate(trajectories):
            print(f"\nProcessing trajectory {i+1}")
            # Tokenize the response
            response_tokens = self.tokenizer(response, return_tensors="pt").to(self.device)
            print(f"Response tokens device: {response_tokens['input_ids'].device}")
            
            # Get model outputs
            print("Getting model outputs...")
            with torch.no_grad():
                outputs = self.model(**response_tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Get last layer hidden states
                print(f"Hidden states shape: {hidden_states.shape}")
            
            # Get policy logits
            print("Calculating policy logits...")
            policy_logits = self.policy_head(hidden_states)
            print(f"Policy logits shape: {policy_logits.shape}")
            
            # Calculate PPO loss
            print("Calculating loss...")
            loss += -reward * torch.log_softmax(policy_logits, dim=-1).mean()
            print(f"Current loss: {loss.item():.4f}")
        
        # Average the loss
        loss = loss / self.num_trajectories
        print(f"\nFinal loss: {loss.item():.4f}")
        
        # Backward pass
        print("Performing backward pass...")
        loss.backward()
        
        # Update policy head
        print("Updating policy head...")
        optimizer = torch.optim.Adam(self.policy_head.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()

class TextEnvironment:
    def __init__(self):
        self.prompts = [
            "Explain the concept of reinforcement learning.",
            "What are the ethical considerations in AI development?",
            "How can we ensure AI systems are aligned with human values?",
            "Describe the difference between supervised and unsupervised learning."
        ]
        self.current_prompt_idx = 0
        
    def reset(self) -> str:
        """Get a new prompt."""
        prompt = self.prompts[self.current_prompt_idx]
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
        return prompt

def train():
    print("Starting training...")
    num_trajectories = 4
    max_steps = 3
    agent = QwenRLAgent(num_trajectories=num_trajectories)
    env = TextEnvironment()
    
    num_episodes = 100
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        prompt = env.reset()
        print(f"Initial Prompt: {prompt}")
        
        # Collect trajectories
        trajectories = agent.generate_trajectories(prompt)
        
        # Calculate average reward for this episode
        episode_reward = np.mean([reward for response, reward in trajectories])
        print(f"\nEpisode Reward: {episode_reward:.4f}")
        
        # Update policy
        loss = agent.train_step(prompt, trajectories[-1][0])
        print(f"Policy Loss: {loss:.4f}")
        
        # Save model if it's the best so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model("reinforced_qwen")
            print(f"\nNew best model saved with reward: {best_reward:.4f}")

if __name__ == "__main__":
    train() 