"""
Zoe Render
CS 370 CartPole Reinforcement Learning Project
Reconstructed from the Module Six pseudocode.

Implements:
1. REINFORCE (Monte Carlo Policy Gradient)
2. Advantage Actor-Critic (A2C-style one-step updates)

Required packages:
    pip install gymnasium[classic-control] torch numpy

Run:
    python cartpole_policy_gradient.py

This file is a reconstruction based on the submitted pseudocode and is not
the original course source file.
"""

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    episodes: int = 300
    gamma: float = 0.99
    learning_rate: float = 0.001
    hidden_size: int = 128
    seed: int = 42
    print_every: int = 20


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(state) -> torch.Tensor:
    """Convert an environment state to a PyTorch tensor."""
    return torch.tensor(state, dtype=torch.float32)


# ---------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Policy network used by REINFORCE.

    Input:
        CartPole state: [cart position, cart velocity,
                         pole angle, pole angular velocity]

    Output:
        Probability of each action.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def calculate_discounted_returns(rewards, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns:

        G_t = r_t + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
    """
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalization usually helps stabilize policy-gradient training.
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def train_reinforce(config: Config):
    """
    Train CartPole using REINFORCE.

    Pseudocode represented:
        Initialize policy network
        For each episode:
            Reset environment
            Store actions/rewards
            For each timestep:
                Choose action from policy
                Observe reward
            Compute discounted return
            Update policy parameters
    """
    print("\n=== Training REINFORCE ===")

    env = gym.make(config.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=config.hidden_size,
    )

    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    episode_scores = []

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)

        log_probabilities = []
        rewards = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_tensor = to_tensor(state)

            action_probabilities = policy(state_tensor)
            action_distribution = Categorical(action_probabilities)
            action = action_distribution.sample()

            log_probabilities.append(action_distribution.log_prob(action))

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            rewards.append(reward)
            state = next_state

        returns = calculate_discounted_returns(rewards, config.gamma)

        policy_loss = []

        for log_probability, discounted_return in zip(
            log_probabilities, returns
        ):
            policy_loss.append(-log_probability * discounted_return)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        score = sum(rewards)
        episode_scores.append(score)

        if episode % config.print_every == 0:
            recent_average = np.mean(episode_scores[-config.print_every:])
            print(
                f"Episode {episode:>3} | "
                f"Score: {score:>5.1f} | "
                f"Recent Avg: {recent_average:>6.2f}"
            )

    env.close()
    return policy, episode_scores


# ---------------------------------------------------------
# A2C
# ---------------------------------------------------------

class ActorNetwork(nn.Module):
    """Actor network: learns the policy pi(a|s)."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class CriticNetwork(nn.Module):
    """Critic network: estimates the state value V(s)."""

    def __init__(self, state_size: int, hidden_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def train_a2c(config: Config):
    """
    Train CartPole using a simple one-step Advantage Actor-Critic approach.

    Pseudocode represented:
        Initialize actor and critic
        For each episode:
            Reset environment
            For each timestep:
                Select action from actor
                Observe reward and next state
                Compute TD error
                Update critic
                Update actor
    """
    print("\n=== Training A2C ===")

    env = gym.make(config.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = ActorNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=config.hidden_size,
    )

    critic = CriticNetwork(
        state_size=state_size,
        hidden_size=config.hidden_size,
    )

    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=config.learning_rate,
    )

    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=config.learning_rate,
    )

    episode_scores = []

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)

        score = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_tensor = to_tensor(state)

            action_probabilities = actor(state_tensor)
            action_distribution = Categorical(action_probabilities)
            action = action_distribution.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_state_tensor = to_tensor(next_state)

            current_value = critic(state_tensor).squeeze()
            next_value = critic(next_state_tensor).squeeze().detach()

            reward_tensor = torch.tensor(reward, dtype=torch.float32)

            if done:
                td_target = reward_tensor
            else:
                td_target = reward_tensor + config.gamma * next_value

            # Advantage / TD error:
            # delta = r + gamma*V(s') - V(s)
            advantage = td_target - current_value

            # Update critic by minimizing squared TD error.
            critic_loss = advantage.pow(2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update actor using the advantage as the learning signal.
            actor_loss = -action_distribution.log_prob(action) * advantage.detach()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            score += reward
            state = next_state

        episode_scores.append(score)

        if episode % config.print_every == 0:
            recent_average = np.mean(episode_scores[-config.print_every:])
            print(
                f"Episode {episode:>3} | "
                f"Score: {score:>5.1f} | "
                f"Recent Avg: {recent_average:>6.2f}"
            )

    env.close()
    return actor, critic, episode_scores


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_policy(policy, env_name: str, episodes: int = 5) -> float:
    """
    Evaluate a policy/actor without training.

    The action with the highest probability is selected.
    """
    env = gym.make(env_name)
    scores = []

    for _ in range(episodes):
        state, _ = env.reset()

        score = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_tensor = to_tensor(state)

            with torch.no_grad():
                probabilities = policy(state_tensor)

            action = torch.argmax(probabilities).item()

            state, reward, terminated, truncated, _ = env.step(action)
            score += reward

        scores.append(score)

    env.close()
    return float(np.mean(scores))


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main():
    config = Config()
    set_seed(config.seed)

    reinforce_policy, reinforce_scores = train_reinforce(config)

    reinforce_average = evaluate_policy(
        reinforce_policy,
        config.env_name,
    )

    print(
        f"\nREINFORCE evaluation average: "
        f"{reinforce_average:.2f}"
    )

    set_seed(config.seed)

    actor, critic, a2c_scores = train_a2c(config)

    a2c_average = evaluate_policy(
        actor,
        config.env_name,
    )

    print(
        f"\nA2C evaluation average: "
        f"{a2c_average:.2f}"
    )


if __name__ == "__main__":
    main()
