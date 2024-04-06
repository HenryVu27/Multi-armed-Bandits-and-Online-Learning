import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=2)
np.random.seed(32)
# Bernoulli reward with parameter theta
def reward(arm, reward_prob):
    return np.random.binomial(n=1, p=reward_prob[arm])

# Beta prior
def TS(num_arms, horizon, reward_prob, best_arm):
    regrets = np.zeros(horizon)
    samples = np.zeros((horizon, num_arms))
    rewards = np.zeros(horizon)
    alphas = np.zeros((horizon+1, num_arms))
    betas = np.zeros((horizon+1, num_arms))
    alpha = np.ones(num_arms)
    beta = np.ones(num_arms)
    for t in range(horizon):
        alphas[t]= alpha
        betas[t] = beta
        for i in range(num_arms):
            samples[t, i] = np.random.beta(alpha[i], beta[i])
        chosen_arm = np.argmax(samples[t,:])
        x = reward(chosen_arm, reward_prob)
        regrets[t] = reward_prob[best_arm] - reward_prob[chosen_arm]
        alpha[chosen_arm] += x
        beta[chosen_arm] += 1-x
        rewards[t] = x
        
    alphas[horizon] = alpha
    betas[horizon] = beta
    
    return samples, alphas, betas, rewards, regrets

reward_prob = [0.9, 0.5, 0.2]
best_arm = 0
num_arms = 3
horizon = 5000

samples, alpha, beta, rewards, regrets = TS(num_arms, horizon, reward_prob, best_arm)

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
plt.figure(figsize=(10, 6))
plt.plot(regrets, label='Thompson Sampling Regret')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Per-step Regret')
plt.title('Regret of Thompson Sampling')