import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=2)
np.random.seed(32)

# Bernoulli reward with parameter theta
def reward(arm, reward_prob):
    return np.random.binomial(n=1, p=reward_prob[arm])

# Beta prior
def TS(num_arms, horizon, alpha, beta, reward_prob):
    samples = np.zeros((horizon, num_arms))
    rewards = np.zeros(horizon)
    for t in range(horizon):
        for i in range(num_arms):
            samples[t, i] = np.random.beta(alpha[t,i], beta[t,i])
        chosen_arm = np.argmax(samples)
        reward = reward(chosen_arm, reward_prob)
        alpha[t+1,i] = alpha[t,i]+reward
        beta[t+1,i] = beta[t,i]+1-reward
        rewards[t] = reward

    return samples, alpha, beta, rewards
reward_prob = [0.1, 0.9, 0.2, 0.3, 0.15]
best_reward = max(reward_prob)
num_arms = 5
horizon = 2000
alpha = np.ones((horizon+1, num_arms))
beta = np.ones((horizon+1, num_arms))

samples, alpha, beta, rewards = TS(num_arms, horizon, alpha, beta, reward_prob)
thetas = alpha/(alpha + beta)
# Cum reward over time
plt.figure(figsize=(10, 6))
for arm in range(num_arms):
    plt.plot(thetas[:, arm], label=f'Arm {arm+1}')
plt.xlabel('Time step')
plt.ylabel('Estimated Reward')
plt.title('Estimated Reward of Each Arm Over Time')
plt.legend()
plt.show()
#-------------------------------------------------------
# Regret over time
ts_sum = np.cumsum(rewards)
ts_regret = (np.arange(horizon)*best_reward - ts_sum)
plt.figure(figsize=(10, 6))
plt.plot(ts_regret, label='Thompson Sampling Regret')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret of Thompson Sampling')
