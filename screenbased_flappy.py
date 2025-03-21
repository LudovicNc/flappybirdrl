"""
3MD3220: Reinforcement Learning - Individual Assignment
Text Flappy Bird Screen-based Implementation with Monte Carlo and Sarsa(λ) agents
"""

import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random
import time

# Import the Text Flappy Bird environment
import text_flappy_bird_gym

# Helper Functions for screen observation processing
def process_screen_observation(screen_obs, height, width):
    """
    Process the screen observation into a flattened binary array
    
    Args:
        screen_obs: The screen observation from the environment
        height: The height of the screen
        width: The width of the screen
        
    Returns:
        np.array: Flattened binary representation of the screen
    """
    # IMPORTANT: Based on error, the screen_obs is of shape (width, height) not (height, width)
    # For TextFlappyBird-screen-v0, the observation is a 2D array of integers (not strings)
    
    # Convert screen to binary array (any non-zero value is an object)
    binary_screen = np.zeros((width, height), dtype=np.int8)
    
    # Check if screen_obs is already a numpy array
    if isinstance(screen_obs, np.ndarray):
        # Convert any non-zero value to 1
        binary_screen = (screen_obs > 0).astype(np.int8)
    else:
        # If it's not a numpy array (e.g., list of strings), iterate through it
        for i in range(min(len(screen_obs), width)):
            for j in range(min(len(screen_obs[i]), height)):
                if screen_obs[i][j] != 0:  # Any non-zero character is an object
                    binary_screen[i, j] = 1
    
    # Return flattened array
    return binary_screen.flatten()

def create_state_buckets(screen_height, screen_width, n_buckets=10):
    """
    Create buckets to discretize the screen state space
    
    Args:
        screen_height: The height of the screen
        screen_width: The width of the screen
        n_buckets: Number of buckets to use
        
    Returns:
        dict: Dictionary with bucket definitions
    """
    # Create buckets for screen regions
    # This divides the screen into regions and tracks occupancy
    h_regions = max(2, n_buckets // 5)  # Horizontal regions
    v_regions = max(2, n_buckets // 2)  # Vertical regions (more important for flappy bird)
    
    h_size = screen_width // h_regions
    v_size = screen_height // v_regions
    
    return {
        'h_regions': h_regions,
        'v_regions': v_regions,
        'h_size': h_size,
        'v_size': v_size
    }

def discretize_screen_state(binary_screen, screen_height, screen_width, buckets):
    """
    Convert binary screen to a discretized state based on region occupancy
    
    Args:
        binary_screen: Flattened binary screen
        screen_height: The height of the screen
        screen_width: The width of the screen
        buckets: Bucket definitions from create_state_buckets
        
    Returns:
        tuple: Discretized state as a tuple of region occupancies
    """
    # Reshape binary screen back to 2D (with correct dimensions)
    screen_2d = binary_screen.reshape((screen_width, screen_height))
    
    # Create state vector based on region occupancy
    state = []
    for v_idx in range(buckets['v_regions']):
        for h_idx in range(buckets['h_regions']):
            # Get region boundaries
            v_start = v_idx * buckets['v_size']
            v_end = min((v_idx + 1) * buckets['v_size'], screen_height)
            h_start = h_idx * buckets['h_size']
            h_end = min((h_idx + 1) * buckets['h_size'], screen_width)
            
            # Check if region contains any objects (1s)
            region_occupancy = np.any(screen_2d[h_start:h_end, v_start:v_end])
            state.append(int(region_occupancy))
    
    return tuple(state)

def plot_value_function_screen(V, buckets, screen_height, screen_width, title="Screen State-Value Function"):
    """
    Plot the state-value function for screen observations
    
    Args:
        V: The value function dictionary
        buckets: Bucket definitions
        screen_height: The height of the screen
        screen_width: The width of the screen
        title: Plot title
    """
    # Since the state space is too large to visualize completely,
    # we'll create a 2D heatmap showing average values for each region
    region_values = np.zeros((buckets['v_regions'], buckets['h_regions']))
    region_counts = np.zeros((buckets['v_regions'], buckets['h_regions']))
    
    # Extract values per region
    for state, value in V.items():
        if isinstance(state, tuple):
            # For each state, update the corresponding regions
            for i, occupied in enumerate(state):
                if occupied:
                    v_idx = i // buckets['h_regions']
                    h_idx = i % buckets['h_regions']
                    
                    region_values[v_idx, h_idx] += value
                    region_counts[v_idx, h_idx] += 1
    
    # Calculate average values (avoid division by zero)
    region_counts[region_counts == 0] = 1
    avg_values = region_values / region_counts
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_values, cmap='viridis', annot=True, fmt='.2f')
    plt.title(title)
    plt.xlabel('Horizontal Region')
    plt.ylabel('Vertical Region')
    plt.show()

def compare_performance(results_dict, title="Algorithm Performance Comparison"):
    """Compare performance of different algorithms"""
    plt.figure(figsize=(12, 6))
    
    for name, returns in results_dict.items():
        # Calculate running average with window=10
        running_avg = np.convolve(returns, np.ones(10)/10, mode='valid')
        plt.plot(running_avg, label=f"{name}")
    
    plt.xlabel('Episode')
    plt.ylabel('Return (Running Average)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Monte Carlo Agent for screen observations (using dictionary for Q-table)
class MonteCarloScreenAgent:
    """On-policy First-Visit Monte Carlo Control for screen observations"""
    def __init__(self, action_space, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = action_space.n
        
        # Use dictionaries for Q-values and returns count due to large state space
        self.Q = {}
        self.returns_count = {}
        
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, defaulting to 0"""
        return self.Q.get((state, action), 0.0)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            # Find action with highest Q-value
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)
    
    def update(self, episode):
        """Update policy after each episode using first-visit MC"""
        # Calculate returns for each state-action pair
        G = 0
        state_action_seen = set()
        
        # Process episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            
            # Calculate return (discounted sum of rewards)
            G = self.gamma * G + reward
            
            # Only update if this is first visit to state-action pair
            if (state, action) not in state_action_seen:
                state_action_seen.add((state, action))
                
                # Get current count or initialize to 0
                count = self.returns_count.get((state, action), 0)
                self.returns_count[(state, action)] = count + 1
                
                # Get current Q-value or initialize to 0
                old_q = self.get_q_value(state, action)
                
                # Update Q-value with incremental mean
                self.Q[(state, action)] = old_q + (G - old_q) / self.returns_count[(state, action)]
    
    def get_value_function(self):
        """Compute state-value function from action-value function"""
        V = {}
        # Group Q-values by state
        state_actions = {}
        for (state, action), q_value in self.Q.items():
            if state not in state_actions:
                state_actions[state] = []
            state_actions[state].append((action, q_value))
        
        # For each state, find max Q-value
        for state, action_values in state_actions.items():
            if action_values:
                V[state] = max(q for _, q in action_values)
            else:
                V[state] = 0.0
        
        return V

# Sarsa(λ) Agent for screen observations (using dictionary for Q-table)
class SarsaLambdaScreenAgent:
    """Sarsa(λ) with eligibility traces for screen observations"""
    def __init__(self, action_space, gamma=0.99, alpha=0.3, epsilon=0.1, lambda_=0.9):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.n_actions = action_space.n
        
        # Use dictionaries for Q-values and eligibility traces due to large state space
        self.Q = {}
        self.E = {}
        
        # Track visited states for more efficient updates
        self.visited_states = set()
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, defaulting to optimistic value"""
        return self.Q.get((state, action), 5.0)  # Optimistic initialization
    
    def get_e_value(self, state, action):
        """Get eligibility trace value for a state-action pair, defaulting to 0"""
        return self.E.get((state, action), 0.0)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            # Find action with highest Q-value
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, next_action, done):
        """Update Q-values using Sarsa(λ) algorithm with eligibility traces"""
        # Add current state to visited states
        self.visited_states.add(state)
        
        # Calculate TD error
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.get_q_value(next_state, next_action)
        
        delta = target - self.get_q_value(state, action)
        
        # Update eligibility trace for current state-action pair (replacing traces)
        self.E[(state, action)] = 1.0
        
        # Update Q-values for all state-action pairs with non-zero eligibility
        for (s, a), e_value in list(self.E.items()):
            # Skip if eligibility is too small
            if e_value < 0.01:
                del self.E[(s, a)]
                continue
                
            # Update Q-value
            old_q = self.get_q_value(s, a)
            self.Q[(s, a)] = old_q + self.alpha * delta * e_value
            
            # Decay eligibility trace
            self.E[(s, a)] *= self.gamma * self.lambda_
        
        # Reset eligibility traces if episode ends
        if done:
            self.E = {}
            self.visited_states = set()
    
    def get_value_function(self):
        """Compute state-value function from action-value function"""
        V = {}
        # Group Q-values by state
        state_actions = {}
        for (state, action), q_value in self.Q.items():
            if state not in state_actions:
                state_actions[state] = []
            state_actions[state].append((action, q_value))
        
        # For each state, find max Q-value
        for state, action_values in state_actions.items():
            if action_values:
                V[state] = max(q for _, q in action_values)
            else:
                V[state] = 0.0
        
        return V

# Training Functions
def train_monte_carlo_screen(env, agent, n_episodes=1000, screen_height=15, screen_width=20, buckets=None):
    """Train Monte Carlo agent with screen observations"""
    episode_returns = []
    
    for episode in tqdm(range(n_episodes), desc="Training Monte Carlo Screen"):
        episode_data = []
        state, _ = env.reset()
        binary_screen = process_screen_observation(state, screen_height, screen_width)
        state_disc = discretize_screen_state(binary_screen, screen_height, screen_width, buckets)
        done = False
        total_reward = 0
        
        # Generate episode
        while not done:
            action = agent.select_action(state_disc)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_data.append((state_disc, action, reward))
            
            state = next_state
            binary_screen = process_screen_observation(state, screen_height, screen_width)
            state_disc = discretize_screen_state(binary_screen, screen_height, screen_width, buckets)
        
        # Update agent using complete episode
        agent.update(episode_data)
        episode_returns.append(total_reward)
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Print progress occasionally
        if episode % 100 == 0 and episode > 0:
            avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            print(f"Episode {episode}: Avg Return = {avg_return:.2f}, Epsilon = {agent.epsilon:.3f}")
        
    return episode_returns

def train_sarsa_lambda_screen(env, agent, n_episodes=1000, screen_height=15, screen_width=20, buckets=None):
    """Train Sarsa(λ) agent with screen observations"""
    episode_returns = []
    
    for episode in tqdm(range(n_episodes), desc="Training Sarsa(λ) Screen"):
        state, _ = env.reset()
        binary_screen = process_screen_observation(state, screen_height, screen_width)
        state_disc = discretize_screen_state(binary_screen, screen_height, screen_width, buckets)
        action = agent.select_action(state_disc)
        done = False
        total_reward = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_binary_screen = process_screen_observation(next_state, screen_height, screen_width)
            next_state_disc = discretize_screen_state(next_binary_screen, screen_height, screen_width, buckets)
            next_action = agent.select_action(next_state_disc) if not done else 0
            
            agent.update(state_disc, action, reward, next_state_disc, next_action, done)
            
            state = next_state
            state_disc = next_state_disc
            action = next_action
            total_reward += reward
        
        episode_returns.append(total_reward)
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Print progress occasionally
        if episode % 100 == 0 and episode > 0:
            avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            print(f"Episode {episode}: Avg Return = {avg_return:.2f}, Epsilon = {agent.epsilon:.3f}")
        
    return episode_returns

# Evaluation Function
def evaluate_agent_screen(env, agent, n_episodes=10, screen_height=15, screen_width=20, buckets=None):
    """Evaluate a trained agent with screen observations"""
    total_returns = []
    total_steps = []
    
    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        state, _ = env.reset()
        binary_screen = process_screen_observation(state, screen_height, screen_width)
        state_disc = discretize_screen_state(binary_screen, screen_height, screen_width, buckets)
        done = False
        episode_return = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state_disc)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            steps += 1
            
            state = next_state
            binary_screen = process_screen_observation(state, screen_height, screen_width)
            state_disc = discretize_screen_state(binary_screen, screen_height, screen_width, buckets)
        
        total_returns.append(episode_return)
        total_steps.append(steps)
    
    return np.mean(total_returns), np.mean(total_steps)

# Parameter Sweep Function
def parameter_sweep_screen(param_name, param_values, env_creator, agent_class, train_func, 
                         n_episodes=500, runs=2, screen_height=15, screen_width=20, buckets=None):
    """Perform parameter sweep for screen-based agents"""
    results = []
    
    for param_value in param_values:
        for run in range(runs):
            # Create environment and agent
            env = env_creator()
            
            # Set the parameter
            kwargs = {param_name: param_value}
            agent = agent_class(
                action_space=env.action_space,
                **kwargs
            )
            
            # Train agent
            returns = train_func(env, agent, n_episodes=n_episodes, 
                                screen_height=screen_height, screen_width=screen_width, buckets=buckets)
            
            # Record results
            for episode, ret in enumerate(returns):
                results.append({
                    'Parameter': param_value,
                    'Run': run,
                    'Episode': episode,
                    'Return': ret
                })
            
            env.close()
    
    return pd.DataFrame(results)

# Environment creation
def create_environment_screen(height=15, width=20, pipe_gap=4):
    """Create the Text Flappy Bird screen environment"""
    env = gym.make('TextFlappyBird-screen-v0', 
                  height=height, 
                  width=width, 
                  pipe_gap=pipe_gap)
    return env

# Main experiment for screen version
def main_screen():
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Environment parameters - IMPORTANT: In TextFlappyBird-screen-v0, 
    # the screen dimensions are reversed from what you might expect
    screen_width = 20  # This corresponds to the first dimension in the observation space
    screen_height = 15  # This corresponds to the second dimension in the observation space
    pipe_gap = 4
    
    # Check environment properties
    env = create_environment_screen(height=screen_height, width=screen_width, pipe_gap=pipe_gap)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Get some example observations
    obs, _ = env.reset()
    print(f"Example observation shape: {obs.shape}")
    
    # Take a few random actions to understand the environment
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Observation shape: {obs.shape}")
        if i == 0:  # Print the first observation to see the screen format
            print("Example screen observation (first few rows):")
            print(obs[:5, :5])  # Print top-left 5x5 corner as example
    
    # Create state discretization buckets (divides screen into regions)
    buckets = create_state_buckets(screen_height, screen_width, n_buckets=20)
    
    # Create Monte Carlo agent for screen observations
    mc_agent = MonteCarloScreenAgent(
        action_space=env.action_space,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Create Sarsa(λ) agent for screen observations
    sarsa_agent = SarsaLambdaScreenAgent(
        action_space=env.action_space,
        gamma=0.99,
        alpha=0.3,
        epsilon=0.1,
        lambda_=0.9
    )
    
    # Train agents (using fewer episodes due to increased complexity)
    print("Training Monte Carlo agent...")
    mc_returns = train_monte_carlo_screen(env, mc_agent, n_episodes=500, 
                                         screen_height=screen_height, 
                                         screen_width=screen_width,
                                         buckets=buckets)
    
    print("\nTraining Sarsa(λ) agent...")
    sarsa_returns = train_sarsa_lambda_screen(env, sarsa_agent, n_episodes=500,
                                             screen_height=screen_height,
                                             screen_width=screen_width, 
                                             buckets=buckets)
    
    # Compare performance
    compare_performance({
        'Monte Carlo Screen': mc_returns,
        'Sarsa(λ) Screen': sarsa_returns
    })
    
    # Plot value functions
    plot_value_function_screen(mc_agent.get_value_function(), buckets, 
                              screen_height, screen_width, 
                              title="Monte Carlo Screen State-Value Function")
    
    plot_value_function_screen(sarsa_agent.get_value_function(), buckets,
                              screen_height, screen_width,
                              title="Sarsa(λ) Screen State-Value Function")
    
    # Parameter sweep for Sarsa(λ)
    env_creator = lambda: create_environment_screen(height=screen_height, width=screen_width, pipe_gap=pipe_gap)
    
    lambda_values = [0.0, 0.5, 0.9, 0.99]
    print("\nPerforming parameter sweep for λ values...")
    lambda_results = parameter_sweep_screen(
        'lambda_',
        lambda_values,
        env_creator,
        SarsaLambdaScreenAgent,
        train_sarsa_lambda_screen,
        n_episodes=200,  # Reduced for time
        runs=2,          # Reduced for time
        screen_height=screen_height,
        screen_width=screen_width,
        buckets=buckets
    )
    
    # Plot parameter sweep results
    plt.figure(figsize=(12, 6))
    for param_value in lambda_values:
        subset = lambda_results[lambda_results['Parameter'] == param_value]
        avg_returns = subset.groupby('Episode')['Return'].mean()
        running_avg = np.convolve(avg_returns, np.ones(10)/10, mode='valid')
        plt.plot(running_avg, label=f"λ={param_value}")
    
    plt.xlabel('Episode')
    plt.ylabel('Return (Running Average)')
    plt.title('Effect of λ on Sarsa(λ) Screen Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Generalization testing
    print("\nEvaluating agents on original environment configuration...")
    mc_avg_return, mc_avg_steps = evaluate_agent_screen(env, mc_agent, n_episodes=10,
                                                      screen_height=screen_height,
                                                      screen_width=screen_width,
                                                      buckets=buckets)
    
    sarsa_avg_return, sarsa_avg_steps = evaluate_agent_screen(env, sarsa_agent, n_episodes=10,
                                                            screen_height=screen_height,
                                                            screen_width=screen_width,
                                                            buckets=buckets)
    
    print(f"Monte Carlo Screen: Avg Return = {mc_avg_return:.2f}, Avg Steps = {mc_avg_steps:.2f}")
    print(f"Sarsa(λ) Screen: Avg Return = {sarsa_avg_return:.2f}, Avg Steps = {sarsa_avg_steps:.2f}")
    
    # Create environment with different configuration
    modified_height = 18
    modified_width = 25
    modified_pipe_gap = 3
    
    env_modified = create_environment_screen(height=modified_height, 
                                           width=modified_width, 
                                           pipe_gap=modified_pipe_gap)
    
    # Create new buckets for modified environment
    modified_buckets = create_state_buckets(modified_height, modified_width, n_buckets=20)
    
    print("\nEvaluating agents on modified environment configuration...")
    mc_avg_return_mod, mc_avg_steps_mod = evaluate_agent_screen(env_modified, mc_agent, n_episodes=10,
                                                              screen_height=modified_height,
                                                              screen_width=modified_width,
                                                              buckets=modified_buckets)
    
    sarsa_avg_return_mod, sarsa_avg_steps_mod = evaluate_agent_screen(env_modified, sarsa_agent, n_episodes=10,
                                                                    screen_height=modified_height,
                                                                    screen_width=modified_width,
                                                                    buckets=modified_buckets)
    
    print(f"Monte Carlo Screen: Avg Return = {mc_avg_return_mod:.2f}, Avg Steps = {mc_avg_steps_mod:.2f}")
    print(f"Sarsa(λ) Screen: Avg Return = {sarsa_avg_return_mod:.2f}, Avg Steps = {sarsa_avg_steps_mod:.2f}")
    
    # Calculate performance difference as a percentage
    if mc_avg_return != 0:
        mc_performance_diff = ((mc_avg_return_mod - mc_avg_return) / mc_avg_return) * 100
        print(f"Monte Carlo Screen performance change: {mc_performance_diff:.2f}%")
    else:
        print("Monte Carlo Screen performance change: N/A (division by zero)")
        
    if sarsa_avg_return != 0:
        sarsa_performance_diff = ((sarsa_avg_return_mod - sarsa_avg_return) / sarsa_avg_return) * 100
        print(f"Sarsa(λ) Screen performance change: {sarsa_performance_diff:.2f}%")
    else:
        print("Sarsa(λ) Screen performance change: N/A (division by zero)")
    
    # Clean up
    env.close()
    env_modified.close()

if __name__ == "__main__":
    main_screen()