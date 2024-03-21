import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=2)
np.random.seed(32)
# NOT YET DONE, THIS CODE IS NOT THE CORRECT IMPLEMENTATION OF EXP4
# using bernoulli reward for simplicity
def reward(arm, reward_prob):
    return np.random.binomial(n=1, p=reward_prob[arm])


def exp4(num_arms, num_experts, horizon, lr):
    cum_reward = np.zeros(num_arms)
    plot = np.zeros((horizon, num_arms))
    rewards = np.zeros((horizon, num_arms))
    act_reward = np.zeros(horizon)
    probs = np.zeros((horizon, num_arms))
    p_t = np.full(num_arms, 1/num_arms)
    for t in range(horizon):
        # update prob
        p_t = np.exp(lr*cum_reward)/np.sum(np.exp(lr*cum_reward))
        # choose arm A_t
        arm = np.random.choice(num_arms, p=p_t)
        indicator = np.zeros(num_arms)
        indicator[arm] = 1
        # receive X_t
        X_t = reward(arm, reward_prob)
        estimated_reward = 1 - (indicator*(1 - X_t))/p_t
        if t <= 10:
            print("Run", t)
            print("arm =", arm)
            print("X_t =", X_t)
            print("prob =", p_t)
            print("reward", estimated_reward)
            print("-------")
        # update the sum reward for all arms
        cum_reward += estimated_reward
        # for plotting
        probs[t] = p_t
        act_reward[t] = X_t
        rewards[t] = estimated_reward
        plot[t] = cum_reward
        
    return plot, rewards, probs, act_reward
reward_prob = [0.1, 0.9, 0.2, 0.3, 0.15]
best_reward = max(reward_prob)
num_arms = 5
num_experts = 3
horizon = 1500
learning_rate = np.sqrt(2*np.log(num_arms)/(horizon*num_arms))
creward, rewards, probs, act_reward = exp4(num_arms, num_experts, horizon, learning_rate)
# Cum reward over time
plt.figure(figsize=(10, 6))
for arm in range(num_arms):
    plt.plot(creward[:, arm], label=f'Arm {arm+1}')
plt.xlabel('Time step')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward of Each Arm Over Time')
print(f"Estimated cummulative reward of each arm at the end of round {horizon}: ", creward[horizon-1])
plt.legend()
#-------------------------------------------------------
# Reward over time
plt.figure(figsize=(10, 6))
for arm in range(num_arms):
    plt.plot(rewards[:, arm], label=f'Arm {arm+1}')
plt.xlabel('Time step')
plt.ylabel('Estimated Reward')
plt.title('Estimated Reward of Each Arm Over Time')
plt.legend()
plt.show()
print(f"Estimated reward of each arm at the end of round {horizon}: ", rewards[horizon-1])
#-------------------------------------------------------
# Weight over time
plt.figure(figsize=(10, 6))
for arm in range(num_arms):
    plt.plot(probs[:, arm], label=f'Arm {arm+1}')
plt.xlabel('Time step')
plt.ylabel('Weight')
plt.title('Weight of Each Arm Over Time')
plt.legend()
plt.show()
print(f"Estimated weight of each arm at the end of round {horizon}: ", probs[horizon-1])
#-------------------------------------------------------
# Regret over time
exp3_sum = np.cumsum(act_reward)
exp3_regret = (np.arange(horizon)*best_reward - exp3_sum)
plt.figure(figsize=(10, 6))
plt.plot(exp3_regret, label='Exp3 Regret')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret of Exp3 Algorithm')
#-------------------------------------------------------
exp3_subregret = (np.arange(horizon)*best_reward - exp3_sum)/np.arange(1,horizon+1)
plt.figure(figsize=(10, 6))
plt.plot(exp3_subregret, label='R_n/n')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('R_n/n')
plt.title('Sublinearity Regret of Exp3 Algorithm')