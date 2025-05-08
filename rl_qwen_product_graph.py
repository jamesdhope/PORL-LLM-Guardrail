import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Dict
import gymnasium as gym
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import os

@dataclass
class Trajectory:
    states: List[str]  # Sequence of states (prompts + responses)
    actions: List[str]  # Sequence of generated responses
    rewards: List[float]  # Sequence of rewards
    log_probs: List[torch.Tensor]  # Sequence of log probabilities

class ProductEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Define graph nodes for competitive products and our products
        self.nodes = {
            # Competitive Products
            "competitor_cloud": "Google Cloud Platform, Microsoft Azure, cloud computing alternatives",
            "competitor_analytics": "Tableau, Power BI, Looker, business intelligence tools, data visualization platforms",
            "competitor_security": "CrowdStrike, Palo Alto Networks, Fortinet, cybersecurity solutions",
            
            # Our Products
            "our_cloud": "Our enterprise cloud platform, scalable infrastructure, managed services, cloud solutions",
            "our_analytics": "Our business intelligence suite, real-time analytics, predictive modeling, data insights",
            "our_security": "Our advanced security platform, threat detection, compliance management, security solutions",
            
            # Transition Nodes
            "pain_points": "cost concerns, scalability issues, security vulnerabilities, integration challenges",
            "benefits": "cost savings, improved performance, enhanced security, better integration",
            "use_cases": "enterprise deployment, small business solutions, industry-specific applications"
        }
        
        # Define valid node transitions as pairs
        self.valid_transitions = [
            # Competitive to Pain Points
            ("competitor_cloud", "pain_points"),
            ("competitor_analytics", "pain_points"),
            ("competitor_security", "pain_points"),
            
            # Pain Points to Benefits
            ("pain_points", "benefits"),
            
            # Benefits to Our Products
            ("benefits", "our_cloud"),
            ("benefits", "our_analytics"),
            ("benefits", "our_security"),
            
            # Our Products to Use Cases
            ("our_cloud", "use_cases"),
            ("our_analytics", "use_cases"),
            ("our_security", "use_cases"),
            
            # Use Cases back to Our Products (for cross-selling)
            ("use_cases", "our_cloud"),
            ("use_cases", "our_analytics"),
            ("use_cases", "our_security")
        ]
        
        # Pre-compute node embeddings
        self.node_embeddings = self._compute_node_embeddings()
    
    def _compute_node_embeddings(self) -> Dict[str, torch.Tensor]:
        """Compute and store embeddings for all nodes."""
        node_embeddings = {}
        for node, description in self.nodes.items():
            inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                node_embeddings[node] = embedding
        return node_embeddings
    
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
    
    def detect_node(self, text: str) -> Tuple[str, float]:
        """Detect which node the text belongs to and its similarity score."""
        text_embedding = self.get_embedding(text)
        max_similarity = -1
        best_node = None
        
        for node, node_embedding in self.node_embeddings.items():
            similarity = F.cosine_similarity(text_embedding, node_embedding, dim=1).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_node = node
                
        return best_node, max_similarity
    
    def check_transition(self, source_node: str, target_node: str) -> float:
        """Check if a transition between nodes is valid."""
        return 1.0 if (source_node, target_node) in self.valid_transitions else 0.0

class RewardModel:
    def __init__(self):
        self.topic_model = ProductEmbeddingModel()
        self.node_sequence = []  # Track the entire sequence of nodes
        self.exploration_factor = 0.8  # Probability of exploring a different node
        
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward based on graph traversal."""
        # Detect current node from response
        current_node, node_similarity = self.topic_model.detect_node(response)
        
        # With some probability, force exploration to a different node
        if len(self.node_sequence) > 0 and torch.rand(1).item() < self.exploration_factor:
            # Get all possible nodes
            all_nodes = list(self.topic_model.nodes.keys())
            # Remove current node and previous nodes from options
            available_nodes = [node for node in all_nodes if node != current_node and node not in self.node_sequence]
            if available_nodes:
                # Randomly select a new node to explore
                current_node = available_nodes[torch.randint(0, len(available_nodes), (1,)).item()]
                print(f"\nForcing exploration to node: {current_node}")
        
        if not self.node_sequence:
            # First response, only reward for being in a valid node
            self.node_sequence.append(current_node)
            return node_similarity
        
        # Calculate transition rewards for all consecutive pairs
        transition_rewards = []
        for i in range(len(self.node_sequence)):
            transition_valid = self.topic_model.check_transition(self.node_sequence[i], current_node)
            transition_rewards.append(transition_valid)
        
        # Add current node to sequence
        self.node_sequence.append(current_node)
        
        # Calculate average transition reward
        avg_transition_reward = sum(transition_rewards) / len(transition_rewards) if transition_rewards else 0.0
        
        # Additional reward for reaching our products
        product_reward = 0.0
        if current_node.startswith("our_"):
            product_reward = 0.5
        
        # Combine rewards (0.5 for node similarity, 0.3 for transition reward, 0.2 for product reward)
        reward = 0.5 * node_similarity + 0.3 * avg_transition_reward + 0.2 * product_reward
        
        return reward
    
    def reset_sequence(self):
        """Reset the node sequence for a new trajectory."""
        self.node_sequence = []

class QwenRLAgent:
    def __init__(self, model_name: str = "Qwen/Qwen-1_8B", num_trajectories: int = 4, max_tokens: int = 200):
        print(f"Initializing QwenRLAgent with model: {model_name}")
        self.device = "cpu"  # Force CPU usage
        print(f"Using device: {self.device}")
        
        self.max_tokens = max_tokens
        print(f"Maximum tokens per response: {max_tokens}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Force CPU usage
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 instead of float16
            low_cpu_mem_usage=True  # Optimize memory usage
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
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the policy head
        torch.save(self.policy_head.state_dict(), os.path.join(path, "policy_head.pt"))
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save the model config
        self.model.config.save_pretrained(path)
        
        print(f"Model saved to {path}")

    def generate_response(self, prompt: str) -> str:
        print(f"\nGenerating response for prompt: {prompt[:50]}...")
        
        # Add system prompt to guide the model
        system_prompt = """You are a professional product recommendation assistant. Your goal is to guide customers from their current solutions to our products through a natural conversation flow.

Follow these guidelines strictly:
1. Always maintain a professional, solution-focused tone
2. Keep responses concise (2-3 sentences maximum)
3. Focus on one topic at a time
4. Never ask multiple questions in one response
5. Never generate nonsensical or off-topic responses
6. Follow this conversation flow:
   - First, acknowledge their current solution
   - Then, discuss relevant pain points or challenges
   - Finally, introduce our solution as a better alternative

Current conversation:"""
        
        # Add response constraints based on current node
        current_node = None
        if self.reward_model.node_sequence:
            current_node = self.reward_model.node_sequence[-1]
            
        if current_node:
            if current_node.startswith("competitor_"):
                system_prompt += "\n\nFocus on understanding their current solution and its limitations."
            elif current_node == "pain_points":
                system_prompt += "\n\nFocus on identifying specific challenges they face with their current solution."
            elif current_node == "benefits":
                system_prompt += "\n\nFocus on how our solution addresses their challenges and provides better value."
            elif current_node.startswith("our_"):
                system_prompt += "\n\nFocus on specific features and advantages of our solution."
            elif current_node == "use_cases":
                system_prompt += "\n\nFocus on relevant use cases and success stories."
        
        full_prompt = f"{system_prompt}\n\nCustomer: {prompt}\nAssistant:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Move inputs to the correct device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        print(f"Input tensor device: {inputs['input_ids'].device}")
        
        # Get current node from previous response if it exists
        if current_node:
            # Get valid transitions from current node
            valid_transitions = [target for source, target in self.reward_model.topic_model.valid_transitions 
                               if source == current_node]
            
            if valid_transitions:
                # Increase temperature and top_p to encourage exploration
                temperature = 0.8 + torch.rand(1).item() * 0.2  # Slightly lower range: 0.8-1.0
                top_p = 0.92 + torch.rand(1).item() * 0.08  # Slightly lower range: 0.92-1.0
                print(f"Encouraging transition from {current_node} to possible nodes: {valid_transitions}")
            else:
                # Normal generation parameters
                temperature = 0.7 + torch.rand(1).item() * 0.2  # 0.7-0.9
                top_p = 0.9 + torch.rand(1).item() * 0.1  # 0.9-1.0
        else:
            # First response, use normal parameters
            temperature = 0.7 + torch.rand(1).item() * 0.2
            top_p = 0.9 + torch.rand(1).item() * 0.1
        
        # Generate response
        print("Generating text...")
        try:
            with torch.no_grad():
                print("Starting generation with parameters:")
                print(f"- max_new_tokens: {self.max_tokens}")
                print(f"- temperature: {temperature:.2f}")
                print(f"- top_p: {top_p:.2f}")
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3  # Prevent repetitive phrases
                )
                print("Generation completed successfully")
                
                # Get the generated text
                full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                
                # Remove the prompt from the response
                response = full_response[len(full_prompt):].strip()
                
                print(f"Generated response length: {len(response)} characters")
                print(f"Generated response: {response[:100]}...")
                return response
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            print("Falling back to CPU...")
            # Try on CPU as fallback
            self.model = self.model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_response[len(full_prompt):].strip()
                return response

    def generate_trajectories(self, prompt: str) -> List[Tuple[str, float]]:
        print(f"\nGenerating {self.num_trajectories} trajectories...")
        trajectories = []
        
        # Reset the node sequence for a new trajectory
        self.reward_model.reset_sequence()
        
        for i in range(self.num_trajectories):
            print(f"\nTrajectory {i+1}/{self.num_trajectories}")
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Calculate reward and get node information
            current_node, node_similarity = self.reward_model.topic_model.detect_node(response)
            reward = self.reward_model.calculate_reward(prompt, response)
            
            # Print node information
            print("\nNode Information:")
            print(f"Current Node: {current_node}")
            print(f"Node Similarity: {node_similarity:.4f}")
            if len(self.reward_model.node_sequence) > 1:
                print(f"Previous Nodes: {self.reward_model.node_sequence[:-1]}")
                print(f"Valid Transitions: {[self.reward_model.topic_model.check_transition(node, current_node) for node in self.reward_model.node_sequence[:-1]]}")
            print(f"Reward: {reward:.4f}")
            
            trajectories.append((response, reward))
            
            # Print the full trajectory so far
            print("\nCurrent Trajectory:")
            for j, (resp, rew) in enumerate(trajectories):
                node, _ = self.reward_model.topic_model.detect_node(resp)
                print(f"Step {j+1}: Node={node}, Reward={rew:.4f}")
                print(f"Response: {resp[:100]}...")
                print("-" * 50)
        
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

class ProductEnvironment:
    def __init__(self):
        self.prompts = [
            "What cloud services do you currently use?",
            "How do you handle your data analytics needs?",
            "What security solutions are you using?",
            "What are your main challenges with your current solutions?",
            "How do you manage your IT infrastructure?",
            "What are your requirements for scalability?",
            "How do you handle compliance requirements?",
            "What's your budget for these services?"
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
    env = ProductEnvironment()
    
    # Create a models directory in the current working directory
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    
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
            # Save to the models directory
            agent.save_model(model_dir)
            print(f"\nNew best model saved with reward: {best_reward:.4f}")

if __name__ == "__main__":
    train() 