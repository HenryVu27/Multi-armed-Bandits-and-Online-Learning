import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.estimated_mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.true_mean

    def update(self, reward):
        self.N += 1
        self.estimated_mean = (1 - 1.0 / self.N) * self.estimated_mean + 1.0 / self.N * reward

# e-greedy
def epsilon_greedy(bandits, N, epsilon=0.1):
    rewards = np.zeros(N)
    for t in range(N):
        if np.random.random() < epsilon:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.estimated_mean for b in bandits])
        reward = bandits[j].pull()
        bandits[j].update(reward)
        rewards[t] = reward
    return rewards

# UCB
def ucb1(bandits, N):
    rewards = np.zeros(N)
    for t in range(N):
        j = np.argmax([b.estimated_mean + np.sqrt(2 * np.log(t + 1) / (b.N + 1)) for b in bandits])
        reward = bandits[j].pull()
        bandits[j].update(reward)
        rewards[t] = reward
    return rewards

# ETC
def explore_then_commit(bandits, N, M):
    rewards = np.zeros(N)
    for t in range(N):
        if t < M:
            # Exploration phase
            j = t % len(bandits)
        else:
            # Commitment phase
            j = np.argmax([b.estimated_mean for b in bandits])
        reward = bandits[j].pull()
        bandits[j].update(reward)
        rewards[t] = reward
    return rewards

true_means = [1.0, 2.0, 3.0, 2.2, 2.5, 5.0, 4.0] 
optimal_mean = max(true_means)
N = 10000  # number of times we pull an arm
M = 1000   # exploration phase for etc

eg_bandits = [Bandit(m) for m in true_means]
ucb_bandits = [Bandit(m) for m in true_means]
etc_bandits = [Bandit(m) for m in true_means]

eg_rewards = epsilon_greedy(eg_bandits, N)
ucb_rewards = ucb1(ucb_bandits, N)
etc_rewards = explore_then_commit(etc_bandits, N, M)

eg_cumulative_rewards = np.cumsum(eg_rewards)
ucb_cumulative_rewards = np.cumsum(ucb_rewards)
etc_cumulative_rewards = np.cumsum(etc_rewards)

eg_regret = np.arange(N) * optimal_mean - eg_cumulative_rewards
ucb_regret = np.arange(N) * optimal_mean - ucb_cumulative_rewards
etc_regret = np.arange(N) * optimal_mean - etc_cumulative_rewards

# plot
plt.figure(figsize=(12, 8))
plt.plot(eg_regret, label='Epsilon-Greedy Regret')
plt.plot(ucb_regret, label='UCB1 Regret')
plt.plot(etc_regret, label='Explore-Then-Commit Regret')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Cumulative Regret')
plt.title('Comparison of Cumulative Regret')
plt.show()
