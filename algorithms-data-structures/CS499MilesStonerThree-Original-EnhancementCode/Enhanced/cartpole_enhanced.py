"""
Zoe Render
08/01/2026
CS 499 Milestone Three - Algorithms and Data Structures Enhancement
Artifact: CS 370 CartPole Reinforcement Learning

This enhanced version builds on the reconstructed CS 370 implementation of
REINFORCE and Advantage Actor-Critic (A2C).

Major enhancements are marked with '# ENHANCEMENT:' comments so the changes
are easy to identify!

Preferred packages:
    pip install gymnasium[classic-control] torch numpy matplotlib

Run:
    python cartpole_enhanced.py

Outputs:
    results/training_results.csv
    results/training_performance.png
    results/summary.txt
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    import gymnasium as gym
except ImportError:
    gym = None

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    episodes: int = 250
    gamma: float = 0.99
    learning_rate: float = 0.001
    hidden_size: int = 128
    seed: int = 42
    print_every: int = 25
    moving_average_window: int = 25
    evaluation_episodes: int = 10
    max_steps: int = 500
    solved_threshold: float = 475.0

    # ENHANCEMENT: Centralized output location so generated evidence is
    # reproducible and easy to include in the ePortfolio submission.
    results_dir: str = "results"


# ---------------------------------------------------------
# Lightweight fallback environment
# ---------------------------------------------------------

class SimpleSpace:
    def __init__(self, n=None, shape=None):
        self.n = n
        self.shape = shape


class SimpleCartPoleEnv:
    """Minimal CartPole-v1 compatible environment used only if Gymnasium is absent."""

    def __init__(self, max_steps: int = 500):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_steps = max_steps
        self.observation_space = SimpleSpace(shape=(4,))
        self.action_space = SimpleSpace(n=2)
        self._rng = np.random.default_rng(42)
        self.state = None
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self._rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self.steps >= self.max_steps
        reward = 1.0
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        pass


def make_env(config: Config):
    # ENHANCEMENT: Added a portability fallback. The preferred path remains
    # Gymnasium, but the project can still be executed for evaluation when
    # the external environment package is unavailable.
    if gym is not None:
        return gym.make(config.env_name)
    return SimpleCartPoleEnv(max_steps=config.max_steps)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(state) -> torch.Tensor:
    return torch.tensor(state, dtype=torch.float32)


def moving_average(values: List[float], window: int) -> List[float]:
    # ENHANCEMENT: Added moving-average calculation to show learning trends
    # instead of relying only on noisy single-episode rewards.
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(float(np.mean(values[start:index + 1])))
    return output


def validate_config(config: Config) -> None:
    # ENHANCEMENT: Added defensive validation so invalid hyperparameters are
    # detected before training begins.
    if config.episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if not 0 < config.gamma <= 1:
        raise ValueError("gamma must be in the range (0, 1]")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be greater than zero")
    if config.moving_average_window <= 0:
        raise ValueError("moving_average_window must be greater than zero")


# ---------------------------------------------------------
# Neural networks
# ---------------------------------------------------------

class PolicyNetwork(nn.Module):
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


class ActorNetwork(PolicyNetwork):
    pass


class CriticNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def calculate_discounted_returns(rewards, gamma: float) -> torch.Tensor:
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ---------------------------------------------------------
# REINFORCE training
# ---------------------------------------------------------

def train_reinforce(config: Config):
    print("\n=== Training REINFORCE ===")
    env = make_env(config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = PolicyNetwork(state_size, action_size, config.hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    # ENHANCEMENT: Store multiple algorithm metrics instead of only score.
    metrics = {"reward": [], "length": [], "loss": []}

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        log_probabilities = []
        rewards = []
        terminated = truncated = False

        while not (terminated or truncated):
            state_tensor = to_tensor(state)
            action_distribution = Categorical(policy(state_tensor))
            action = action_distribution.sample()
            log_probabilities.append(action_distribution.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)

        returns = calculate_discounted_returns(rewards, config.gamma)
        policy_loss = [
            -log_probability * discounted_return
            for log_probability, discounted_return in zip(log_probabilities, returns)
        ]

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        metrics["reward"].append(float(sum(rewards)))
        metrics["length"].append(len(rewards))
        metrics["loss"].append(float(loss.detach().item()))

        if episode % config.print_every == 0:
            recent = np.mean(metrics["reward"][-config.print_every:])
            print(f"Episode {episode:>3} | Reward {metrics['reward'][-1]:>6.1f} | Avg {recent:>7.2f} | Loss {metrics['loss'][-1]:>8.3f}")

    env.close()
    metrics["moving_average"] = moving_average(metrics["reward"], config.moving_average_window)
    return policy, metrics


# ---------------------------------------------------------
# A2C training
# ---------------------------------------------------------

def train_a2c(config: Config):
    print("\n=== Training A2C ===")
    env = make_env(config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    actor = ActorNetwork(state_size, action_size, config.hidden_size)
    critic = CriticNetwork(state_size, config.hidden_size)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.learning_rate)

    # ENHANCEMENT: Track reward, episode length, actor loss, and critic loss
    # for a consistent comparison with REINFORCE.
    metrics = {"reward": [], "length": [], "actor_loss": [], "critic_loss": []}

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        score = 0.0
        episode_actor_losses = []
        episode_critic_losses = []
        steps = 0

        while not (terminated or truncated):
            state_tensor = to_tensor(state)
            action_distribution = Categorical(actor(state_tensor))
            action = action_distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            current_value = critic(state_tensor).squeeze()
            next_value = critic(to_tensor(next_state)).squeeze().detach()
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            td_target = reward_tensor if done else reward_tensor + config.gamma * next_value
            advantage = td_target - current_value

            critic_loss = advantage.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -action_distribution.log_prob(action) * advantage.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            episode_actor_losses.append(float(actor_loss.detach().item()))
            episode_critic_losses.append(float(critic_loss.detach().item()))
            score += reward
            steps += 1
            state = next_state

        metrics["reward"].append(float(score))
        metrics["length"].append(steps)
        metrics["actor_loss"].append(float(np.mean(episode_actor_losses)))
        metrics["critic_loss"].append(float(np.mean(episode_critic_losses)))

        if episode % config.print_every == 0:
            recent = np.mean(metrics["reward"][-config.print_every:])
            print(f"Episode {episode:>3} | Reward {score:>6.1f} | Avg {recent:>7.2f} | ActorLoss {metrics['actor_loss'][-1]:>8.3f}")

    env.close()
    metrics["moving_average"] = moving_average(metrics["reward"], config.moving_average_window)
    return actor, critic, metrics


# ---------------------------------------------------------
# Evaluation and evidence generation
# ---------------------------------------------------------

def evaluate_policy(policy, config: Config) -> Dict[str, float]:
    # ENHANCEMENT: Evaluation now returns average, standard deviation,
    # minimum, maximum, and success rate rather than a single average.
    env = make_env(config)
    scores = []
    for episode in range(config.evaluation_episodes):
        state, _ = env.reset(seed=config.seed + 1000 + episode)
        terminated = truncated = False
        score = 0.0
        while not (terminated or truncated):
            with torch.no_grad():
                probabilities = policy(to_tensor(state))
            action = torch.argmax(probabilities).item()
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
        scores.append(score)
    env.close()
    return {
        "average": float(np.mean(scores)),
        "std_dev": float(np.std(scores)),
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "success_rate": float(np.mean(np.array(scores) >= config.solved_threshold) * 100.0),
    }


def save_training_csv(reinforce_metrics, a2c_metrics, output_path: Path) -> None:
    # ENHANCEMENT: Export episode-level evidence to CSV so results can be
    # independently reviewed instead of existing only in console output.
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "episode",
            "reinforce_reward",
            "reinforce_length",
            "reinforce_loss",
            "reinforce_moving_average",
            "a2c_reward",
            "a2c_length",
            "a2c_actor_loss",
            "a2c_critic_loss",
            "a2c_moving_average",
        ])
        for index in range(len(reinforce_metrics["reward"])):
            writer.writerow([
                index + 1,
                reinforce_metrics["reward"][index],
                reinforce_metrics["length"][index],
                reinforce_metrics["loss"][index],
                reinforce_metrics["moving_average"][index],
                a2c_metrics["reward"][index],
                a2c_metrics["length"][index],
                a2c_metrics["actor_loss"][index],
                a2c_metrics["critic_loss"][index],
                a2c_metrics["moving_average"][index],
            ])


def save_plot(reinforce_metrics, a2c_metrics, output_path: Path) -> None:
    # ENHANCEMENT: Added visual comparison of raw rewards and moving averages
    # so algorithm stability and learning progress can be evaluated directly.
    episodes = np.arange(1, len(reinforce_metrics["reward"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reinforce_metrics["reward"], alpha=0.25, label="REINFORCE reward")
    plt.plot(episodes, a2c_metrics["reward"], alpha=0.25, label="A2C reward")
    plt.plot(episodes, reinforce_metrics["moving_average"], linewidth=2, label="REINFORCE moving avg")
    plt.plot(episodes, a2c_metrics["moving_average"], linewidth=2, label="A2C moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Episode Length")
    plt.title("CartPole Algorithm Training Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_summary(reinforce_eval, a2c_eval, output_path: Path, using_gym: bool) -> None:
    # ENHANCEMENT: Added a concise machine-generated summary for portfolio
    # evidence and instructor review.
    winner = "REINFORCE" if reinforce_eval["average"] > a2c_eval["average"] else "A2C"
    with output_path.open("w", encoding="utf-8") as file:
        file.write("CS 499 CartPole Enhancement Results\n")
        file.write("=================================\n")
        file.write(f"Environment: {'Gymnasium CartPole-v1' if using_gym else 'local CartPole-compatible fallback'}\n\n")
        for name, result in [("REINFORCE", reinforce_eval), ("A2C", a2c_eval)]:
            file.write(f"{name}\n")
            file.write(f"  Evaluation average: {result['average']:.2f}\n")
            file.write(f"  Standard deviation: {result['std_dev']:.2f}\n")
            file.write(f"  Minimum: {result['minimum']:.2f}\n")
            file.write(f"  Maximum: {result['maximum']:.2f}\n")
            file.write(f"  Success rate (>={475}): {result['success_rate']:.1f}%\n\n")
        file.write(f"Higher evaluation average in this run: {winner}\n")
        file.write("Note: reinforcement learning is stochastic, so results vary by run and configuration.\n")


def main():
    config = Config()
    validate_config(config)
    results_dir = Path(__file__).resolve().parent / config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    reinforce_policy, reinforce_metrics = train_reinforce(config)
    reinforce_eval = evaluate_policy(reinforce_policy, config)

    set_seed(config.seed)
    actor, critic, a2c_metrics = train_a2c(config)
    a2c_eval = evaluate_policy(actor, config)

    save_training_csv(reinforce_metrics, a2c_metrics, results_dir / "training_results.csv")
    save_plot(reinforce_metrics, a2c_metrics, results_dir / "training_performance.png")
    write_summary(reinforce_eval, a2c_eval, results_dir / "summary.txt", gym is not None)

    print("\n=== Final Evaluation ===")
    print(f"REINFORCE: {reinforce_eval}")
    print(f"A2C:       {a2c_eval}")
    print(f"\nEvidence saved in: {results_dir}")


if __name__ == "__main__":
    main()
