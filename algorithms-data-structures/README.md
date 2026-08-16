# CartPole Reinforcement Learning Project

<img width="1536" height="1024" alt="ChatGPT Image Aug 16, 2026 at 04_02_50 PM" src="https://github.com/user-attachments/assets/74811a35-5433-4bd6-a6f0-b844654b7672" />


*AI-generated visual created with ChatGPT to represent the CartPole reinforcement-learning enhancement.*

## Algorithms and Data Structures Artifact

This project demonstrates reinforcement learning using the CartPole control problem. It compares two policy-based learning approaches:

* **REINFORCE**, a Monte Carlo policy-gradient algorithm
* **Advantage Actor-Critic (A2C)**, which combines a policy-learning actor with a value-estimating critic

The project focuses on more than simply training an agent. The enhanced version evaluates how the algorithms learn, how consistently they perform, and what trade-offs exist between different reinforcement-learning approaches.


## Artifact Background

The original CS 370 source file was no longer available when this portfolio enhancement was completed. For that reason, the baseline version was reconstructed from retained project documentation and pseudocode.

The reconstructed baseline was preserved so that the original design and the enhanced implementation can be compared directly.

**Original artifact:** `cartpole_policy_gradient.py`

**Enhanced artifact:** `cartpole_enhanced.py`

## CartPole Environment

CartPole is a reinforcement-learning environment in which an agent attempts to balance a pole attached to a moving cart.

The environment provides four state values:

* Cart position
* Cart velocity
* Pole angle
* Pole angular velocity

The agent chooses between two actions:

* Move the cart left
* Move the cart right

The objective is to keep the pole balanced for as long as possible.

The preferred environment for this project is:

`Gymnasium CartPole-v1`

The enhanced version also includes a lightweight CartPole-compatible fallback environment so the training and evaluation logic can still execute when Gymnasium is unavailable.

## REINFORCE

REINFORCE is a Monte Carlo policy-gradient algorithm.

The policy network receives the current CartPole state and produces probabilities for the available actions. During each episode, the algorithm stores rewards and action probabilities.

After the episode ends, discounted returns are calculated and used to update the policy.

This approach is relatively simple, but its updates can be affected by variance because the algorithm waits until the end of an episode before learning from the collected experience.

## Advantage Actor-Critic (A2C)

The A2C implementation uses two neural networks:

* **Actor:** Learns which action should be selected
* **Critic:** Estimates the value of the current state

Unlike REINFORCE, the actor-critic approach updates more frequently during an episode.

The critic estimates how valuable a state is, while the actor updates its policy based on the calculated advantage.

This design introduces additional complexity but provides another way to evaluate reinforcement-learning performance.

## Enhancements

The enhanced version expands the project from basic algorithm execution into a measurable comparison of learning behavior.

Major enhancements include:

* Moving-average reward tracking
* Episode-length tracking
* REINFORCE policy-loss tracking
* A2C actor-loss tracking
* A2C critic-loss tracking
* Consistent evaluation statistics
* Success-rate calculations
* CSV export of episode-level results
* Training-performance visualization
* Centralized results storage
* Hyperparameter validation
* Reproducible seed configuration
* CartPole-compatible fallback environment
* Clearly labeled `ENHANCEMENT` comments throughout the source code

## Configuration Validation

The enhanced project performs validation before training begins.

The program checks important configuration values including:

* Number of episodes
* Discount factor
* Learning rate
* Hidden-layer size
* Moving-average window size

Invalid values raise an error rather than allowing training to continue with an unusable configuration.

## Evaluation

Instead of reporting only one final score, the enhanced project evaluates each trained policy across multiple episodes.

Evaluation includes:

* Average reward
* Standard deviation
* Minimum reward
* Maximum reward
* Success rate

The success threshold is set to **475 points**.

Using repeated evaluation episodes provides stronger evidence of algorithm behavior than relying on a single run.

## Results

During the verified run documented for this enhancement, REINFORCE achieved:

* **Average evaluation reward:** 487.1
* **Evaluation episodes:** 10
* **Success rate:** 90%

A2C achieved:

* **Average evaluation reward:** 53.8

These results should not be interpreted as proof that REINFORCE is always superior to A2C. Reinforcement learning is stochastic, and performance can vary based on hyperparameters, initialization, network design, and training conditions.

The purpose of the enhancement is to make those differences measurable rather than assuming one algorithm should outperform another.

## Generated Output

Running the enhanced program creates a `results` directory containing:

```text
results/
├── training_results.csv
├── training_performance.png
└── summary.txt
```

### `training_results.csv`

Contains episode-level metrics for both algorithms, including rewards, episode lengths, losses, and moving averages.

### `training_performance.png`

Provides a visual comparison of:

* REINFORCE raw rewards
* A2C raw rewards
* REINFORCE moving-average rewards
* A2C moving-average rewards

### `summary.txt`

Contains the final evaluation statistics for both algorithms and identifies which algorithm achieved the higher average reward during that run.

## Requirements

The project requires Python 3.

Recommended packages:

```bash
pip install gymnasium[classic-control] torch numpy matplotlib
```

The enhanced project can use its built-in CartPole-compatible fallback if Gymnasium is unavailable, but PyTorch, NumPy, and Matplotlib are still required.

## Running the Original Artifact

Navigate to the folder containing the original source file and run:

```bash
python3 cartpole_policy_gradient.py
```

The original version trains both REINFORCE and A2C and reports episode scores and basic averages.

## Running the Enhanced Artifact

Navigate to the folder containing the enhanced source file and run:

```bash
python3 cartpole_enhanced.py
```

The program will:

1. Validate the configuration.
2. Train the REINFORCE policy.
3. Evaluate the REINFORCE policy.
4. Train the A2C actor and critic.
5. Evaluate the A2C policy.
6. Export episode-level results.
7. Generate a training comparison graph.
8. Write a summary of the final evaluation.

The generated evidence will be stored automatically in the `results` folder.

## Algorithmic Trade-Offs

This enhancement demonstrates an important concept in computer science: a more complex algorithm does not automatically produce stronger results.

REINFORCE uses a simpler episodic update process, while A2C introduces a critic network and more frequent updates.

The enhanced project makes it possible to compare these approaches using consistent measurements instead of judging them only by whether the program executes successfully.

Moving averages help reduce the influence of noisy individual episodes, while loss tracking and repeated evaluation provide additional information about training behavior.

## Skills Demonstrated

This artifact demonstrates experience with:

* Python
* Reinforcement learning
* Machine learning
* Algorithms
* Data structures
* PyTorch
* Gymnasium
* Neural networks
* Policy-gradient methods
* Actor-critic methods
* Data analysis
* CSV data generation
* Data visualization
* Configuration validation
* Reproducible experimentation
* Algorithm evaluation
* Technical documentation

## Portfolio Outcomes

This project most directly demonstrates the ability to design and evaluate computing solutions using algorithmic principles and appropriate computer science practices.

It also demonstrates the use of modern computing tools to create measurable technical solutions and the ability to communicate algorithmic decisions through documentation, visualization, and analysis.

## Author

**Zoe Render**
Computer Science | Software Engineering
