"""
3MD3220: Reinforcement Learning - Individual Assignment
Text Flappy Bird Implementation with Monte Carlo and Sarsa(λ) agents
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

# Helper Functions
def discretize_state(state, bins):
    """Convert continuous state to discrete state indices for tabular methods"""
    x_dist, y_dist = state
    
    # Ensure the state is within the bin ranges by clipping
    x_dist = np.clip(x_dist, bins['x_bins'][0], bins['x_bins'][-1])
    y_dist = np.clip(y_dist, bins['y_bins'][0], bins['y_bins'][-1])
    
    # Get the bin index
    x_idx = np.digitize(x_dist, bins['x_bins']) - 1  # Subtract 1 to start from 0
    y_idx = np.digitize(y_dist, bins['y_bins']) - 1  # Subtract 1 to start from 0
    
    # Double-check to ensure indices are within bounds
    x_idx = min(x_idx, len(bins['x_bins']) - 2)
    y_idx = min(y_idx, len(bins['y_bins']) - 2)
    
    return (x_idx, y_idx)

def plot_value_function(V, bins, title="State-Value Function"):
    """Plot the state-value function"""
    x_centers = (bins['x_bins'][:-1] + bins['x_bins'][1:]) / 2
    y_centers = (bins['y_bins'][:-1] + bins['y_bins'][1:]) / 2
    
    X, Y = np.meshgrid(x_centers, y_centers)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, V.T, cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel('X Distance to Pipe')
    plt.ylabel('Y Distance to Pipe Center')
    plt.title(title)
    plt.show()

def plot_policy(policy, bins, title="Policy"):
    """Plot the policy (0=do nothing, 1=flap)"""
    # Extract the policy for action 1 (flap)
    if len(policy.shape) == 3:  # If we have a policy for each action
        flap_policy = policy[:,:,1]  # Extract probability of flapping
    else:  # If we just have the actions
        flap_policy = policy
        
    x_centers = (bins['x_bins'][:-1] + bins['x_bins'][1:]) / 2
    y_centers = (bins['y_bins'][:-1] + bins['y_bins'][1:]) / 2
    
    X, Y = np.meshgrid(x_centers, y_centers)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, flap_policy.T, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='Action (0=do nothing, 1=flap)')
    plt.xlabel('X Distance to Pipe')
    plt.ylabel('Y Distance to Pipe Center')
    plt.title(title)
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

# Monte Carlo Agent
class MonteCarloAgent:
    """On-policy First-Visit Monte Carlo Control"""
    def __init__(self, state_space, action_space, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = action_space.n
        
        # Initialize Q-table, returns count, and policy
        self.Q = np.zeros(state_space + (self.n_actions,))
        self.returns_count = np.zeros(state_space + (self.n_actions,))
        self.policy = np.ones(state_space + (self.n_actions,)) / self.n_actions
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, episode):
        """Update policy after each episode using first-visit MC"""
        # Calculate returns for each state-action pair
        G = 0
        state_action_seen = {}
        
        # Process episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            
            # Calculate return (discounted sum of rewards)
            G = self.gamma * G + reward
            
            # Only update if this is first visit to state-action pair
            if (state, action) not in state_action_seen:
                state_action_seen[(state, action)] = True
                
                # Increment returns count for state-action pair
                self.returns_count[state][action] += 1
                
                # Update Q-value with incremental mean
                self.Q[state][action] += (G - self.Q[state][action]) / self.returns_count[state][action]
                
                # Update policy to be epsilon-greedy with respect to Q
                best_action = np.argmax(self.Q[state])
                for a in range(self.n_actions):
                    # Epsilon probability of random action, otherwise greedy
                    self.policy[state][a] = self.epsilon / self.n_actions
                    if a == best_action:
                        self.policy[state][a] += 1 - self.epsilon
    
    def get_value_function(self):
        """Compute state-value function from action-value function"""
        return np.max(self.Q, axis=-1)
    
    def get_policy_function(self):
        """Get the deterministic policy (action with highest Q-value)"""
        return np.argmax(self.Q, axis=-1)

# Sarsa(λ) Agent
class SarsaLambdaAgent:
    """Sarsa(λ) with eligibility traces as described in section 12.7 of the RL textbook"""
    def __init__(self, state_space, action_space, gamma=0.99, alpha=0.3, epsilon=0.1, lambda_=0.9):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.n_actions = action_space.n
        
        # Initialize Q-table and eligibility traces with optimistic initialization
        self.Q = np.ones(state_space + (self.n_actions,)) * 5.0  # Optimistic initialization
        self.E = np.zeros(state_space + (self.n_actions,))
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """Update Q-values using Sarsa(λ) algorithm with eligibility traces"""
        # Calculate TD error
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        
        delta = target - self.Q[state][action]
        
        # Using replacing traces instead of accumulating traces
        self.E[state][action] = 1.0  # Replacing traces
        
        # Update Q-values directly
        self.Q += self.alpha * delta * self.E
        
        # Decay eligibility traces
        self.E *= self.gamma * self.lambda_
        
        # Reset eligibility traces if episode ends
        if done:
            self.E = np.zeros_like(self.E)
    
    def get_value_function(self):
        """Compute state-value function from action-value function"""
        return np.max(self.Q, axis=-1)
    
    def get_policy_function(self):
        """Get the deterministic policy (action with highest Q-value)"""
        return np.argmax(self.Q, axis=-1)

# Training Functions
def train_monte_carlo(env, agent, n_episodes=1000, bins=None):
    """Train Monte Carlo agent"""
    episode_returns = []
    
    for episode in tqdm(range(n_episodes), desc="Training Monte Carlo"):
        episode_data = []
        state, _ = env.reset()
        state_disc = discretize_state(state, bins)
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
            state_disc = discretize_state(state, bins)
        
        # Update agent using complete episode
        agent.update(episode_data)
        episode_returns.append(total_reward)
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Print progress occasionally
        if episode % 100 == 0 and episode > 0:
            avg_return = np.mean(episode_returns[-100:])
            print(f"Episode {episode}: Avg Return = {avg_return:.2f}, Epsilon = {agent.epsilon:.3f}")
        
    return episode_returns

def train_sarsa_lambda(env, agent, n_episodes=1000, bins=None):
    """Train Sarsa(λ) agent"""
    episode_returns = []
    
    for episode in tqdm(range(n_episodes), desc="Training Sarsa(λ)"):
        state, _ = env.reset()
        state_disc = discretize_state(state, bins)
        action = agent.select_action(state_disc)
        done = False
        total_reward = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state_disc = discretize_state(next_state, bins)
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
            avg_return = np.mean(episode_returns[-100:])
            print(f"Episode {episode}: Avg Return = {avg_return:.2f}, Epsilon = {agent.epsilon:.3f}")
        
    return episode_returns

# Evaluation Function
def evaluate_agent(env, agent, bins, n_episodes=10):
    """Evaluate a trained agent"""
    total_returns = []
    total_steps = []
    
    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        state, _ = env.reset()
        state_disc = discretize_state(state, bins)
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
            state_disc = discretize_state(state, bins)
        
        total_returns.append(episode_return)
        total_steps.append(steps)
    
    return np.mean(total_returns), np.mean(total_steps)

# Parameter Sweep Function
def parameter_sweep(param_name, param_values, env_creator, agent_class, train_func, n_episodes=500, runs=3, bins=None):
    """Perform parameter sweep"""
    results = []
    
    for param_value in param_values:
        for run in range(runs):
            # Create environment and agent
            env = env_creator()
            
            # Set the parameter
            kwargs = {param_name: param_value}
            agent = agent_class(
                state_space=(len(bins['x_bins'])-1, len(bins['y_bins'])-1),
                action_space=env.action_space,
                **kwargs
            )
            
            # Train agent
            returns = train_func(env, agent, n_episodes=n_episodes, bins=bins)
            
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
def create_environment(env_type="simple", height=15, width=20, pipe_gap=4):
    """Create the Text Flappy Bird environment"""
    if env_type == "simple":
        env = gym.make('TextFlappyBird-v0', 
                      height=height, 
                      width=width, 
                      pipe_gap=pipe_gap)
    else:
        env = gym.make('TextFlappyBird-screen-v0', 
                      height=height,
                      width=width,
                      pipe_gap=pipe_gap)
    
    return env

# Main experiment
def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Check environment properties
    env = create_environment(env_type="simple")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Get some example observations
    obs, _ = env.reset()
    print(f"Example observation: {obs}")
    
    # Take a few random actions to see how observations change
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, New observation: {obs}, Reward: {reward}")
    
    # Define state discretization based on the observation space
    x_bins = np.linspace(0, 15, 15)   # Covers the range 0 to 14 (x coordinate in the game)
    y_bins = np.linspace(-12, 11, 24) # Covers the range -11 to 10 (y coordinate in the game)
    bins = {'x_bins': x_bins, 'y_bins': y_bins} 
    
    # Create Monte Carlo agent
    mc_agent = MonteCarloAgent(
        state_space=(len(x_bins)-1, len(y_bins)-1),
        action_space=env.action_space,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Create Sarsa(λ) agent
    sarsa_agent = SarsaLambdaAgent(
        state_space=(len(x_bins)-1, len(y_bins)-1),
        action_space=env.action_space,
        gamma=0.99,
        alpha=0.3,
        epsilon=0.1,
        lambda_=0.9
    )
    
    # Train agents
    print("Training Monte Carlo agent...")
    mc_returns = train_monte_carlo(env, mc_agent, n_episodes=1000, bins=bins)
    
    print("\nTraining Sarsa(λ) agent...")
    sarsa_returns = train_sarsa_lambda(env, sarsa_agent, n_episodes=1000, bins=bins)
    
    # Compare performance
    compare_performance({
        'Monte Carlo': mc_returns,
        'Sarsa(λ)': sarsa_returns
    })
    
    # Plot value functions
    plot_value_function(mc_agent.get_value_function(), bins, title="Monte Carlo State-Value Function")
    plot_value_function(sarsa_agent.get_value_function(), bins, title="Sarsa(λ) State-Value Function")
    
    # Plot policies
    plot_policy(mc_agent.get_policy_function(), bins, title="Monte Carlo Policy")
    plot_policy(sarsa_agent.get_policy_function(), bins, title="Sarsa(λ) Policy")
    
    # Parameter sweep for Sarsa(λ)
    env_creator = lambda: create_environment(env_type="simple")
    
    lambda_values = [0.0, 0.5, 0.9, 0.99]
    print("\nPerforming parameter sweep for λ values...")
    lambda_results = parameter_sweep(
        'lambda_',
        lambda_values,
        env_creator,
        SarsaLambdaAgent,
        train_sarsa_lambda,
        n_episodes=300,  # Reduced for time
        runs=2,          # Reduced for time
        bins=bins
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
    plt.title('Effect of λ on Sarsa(λ) Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Generalization testing
    print("\nEvaluating agents on original environment configuration...")
    mc_avg_return, mc_avg_steps = evaluate_agent(env, mc_agent, bins, n_episodes=10)
    sarsa_avg_return, sarsa_avg_steps = evaluate_agent(env, sarsa_agent, bins, n_episodes=10)
    
    print(f"Monte Carlo: Avg Return = {mc_avg_return:.2f}, Avg Steps = {mc_avg_steps:.2f}")
    print(f"Sarsa(λ): Avg Return = {sarsa_avg_return:.2f}, Avg Steps = {sarsa_avg_steps:.2f}")
    
    # Create environment with different configuration
    env_modified = create_environment(env_type="simple", height=18, width=25, pipe_gap=3)
    
    print("\nEvaluating agents on modified environment configuration...")
    mc_avg_return_mod, mc_avg_steps_mod = evaluate_agent(env_modified, mc_agent, bins, n_episodes=10)
    sarsa_avg_return_mod, sarsa_avg_steps_mod = evaluate_agent(env_modified, sarsa_agent, bins, n_episodes=10)
    
    print(f"Monte Carlo: Avg Return = {mc_avg_return_mod:.2f}, Avg Steps = {mc_avg_steps_mod:.2f}")
    print(f"Sarsa(λ): Avg Return = {sarsa_avg_return_mod:.2f}, Avg Steps = {sarsa_avg_steps_mod:.2f}")
    
    # Calculate performance difference as a percentage
    mc_performance_diff = ((mc_avg_return_mod - mc_avg_return) / mc_avg_return) * 100
    sarsa_performance_diff = ((sarsa_avg_return_mod - sarsa_avg_return) / sarsa_avg_return) * 100
    
    print(f"Monte Carlo performance change: {mc_performance_diff:.2f}%")
    print(f"Sarsa(λ) performance change: {sarsa_performance_diff:.2f}%")
    
    # Clean up
    env.close()
    env_modified.close()

if __name__ == "__main__":
    main()