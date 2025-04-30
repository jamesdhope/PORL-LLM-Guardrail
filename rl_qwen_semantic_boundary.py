import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F

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
            embeddings = []
            for desc in descriptions:
                inputs = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                    embeddings.append(embedding)
            
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
        similarities = []
        for topic_embedding in self.topic_embeddings.values():
            similarity = F.cosine_similarity(text_embedding, topic_embedding, dim=1)
            similarities.append(similarity.item())
        return max(similarities)

class RewardModel:
    def __init__(self):
        self.topic_model = TopicEmbeddingModel()
        
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward based on topic similarity."""
        topic_similarity = self.topic_model.calculate_topic_similarity(response)
        prompt_embedding = self.topic_model.get_embedding(prompt)
        response_embedding = self.topic_model.get_embedding(response)
        coherence = F.cosine_similarity(prompt_embedding, response_embedding, dim=1).item()
        reward = 0.7 * topic_similarity + 0.3 * coherence
        return reward

class QwenRLAgent:
    def __init__(self, model_name: str = "Qwen/Qwen-1_8B"):
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
        torch.save(self.policy_head.state_dict(), f"{path}/policy_head.pt")
        self.tokenizer.save_pretrained(path)
        self.model.config.save_pretrained(path)
        print(f"Model saved to {path}")

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                return response[len(prompt):]
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            print("Falling back to CPU...")
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

    def train_step(self, prompt: str, target_response: str) -> float:
        """Simple policy gradient training step."""
        # Generate a single response
        response = self.generate_response(prompt)
        
        # Calculate reward
        reward = self.reward_model.calculate_reward(prompt, response)
        
        # Get model outputs for policy gradient
        response_tokens = self.tokenizer(response, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**response_tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Get last layer hidden states
            
        # Calculate policy gradient loss
        policy_logits = self.policy_head(hidden_states)
        
        # Calculate policy gradient loss (using log probabilities)
        loss = -reward * torch.log_softmax(policy_logits, dim=-1).mean()
        
        # Update policy head
        optimizer = torch.optim.Adam(self.policy_head.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

class TextEnvironment:
    def __init__(self):
        self.prompts = [
            "Explain the concept of reinforcement learning.",
            "What are the ethical considerations in AI?",
            "Describe the process of machine learning model training."
        ]
        self.current_prompt_idx = 0
    
    def reset(self) -> str:
        """Reset the environment and return the first prompt."""
        self.current_prompt_idx = 0
        return self.prompts[self.current_prompt_idx]
    
    def step(self, response: str) -> Tuple[str, float, bool]:
        """Take a step in the environment."""
        prompt = self.prompts[self.current_prompt_idx]
        reward = self.calculate_reward(prompt, response)
        
        self.current_prompt_idx += 1
        done = self.current_prompt_idx >= len(self.prompts)
        
        if not done:
            next_prompt = self.prompts[self.current_prompt_idx]
        else:
            next_prompt = None
            
        return next_prompt, reward, done
    
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward for the response."""
        reward_model = RewardModel()
        return reward_model.calculate_reward(prompt, response)

def train():
    """Train the RL agent."""
    agent = QwenRLAgent()
    env = TextEnvironment()
    
    num_episodes = 10
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        prompt = env.reset()
        total_reward = 0
        
        while True:
            # Generate response
            response = agent.generate_response(prompt)
            
            # Get next state and reward
            next_prompt, reward, done = env.step(response)
            total_reward += reward
            
            # Train on the current step
            loss = agent.train_step(prompt, response)
            print(f"Step reward: {reward:.4f}, Loss: {loss:.4f}")
            
            if done:
                break
                
            prompt = next_prompt
        
        print(f"Episode {episode + 1} completed. Total reward: {total_reward:.4f}")
    
    # Save the trained model
    agent.save_model("trained_model")

if __name__ == "__main__":
    train() 