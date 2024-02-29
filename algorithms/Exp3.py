import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=2)


# using binary reward for simplicity
def reward(arm, reward_prob):
    return np.random.binomial(n=1, p=reward_prob[arm])


def exp3(num_arms, horizon, lr):
    cum_reward = np.zeros(num_arms)
    plot = np.zeros((horizon, num_arms))
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
        plot[t] = cum_reward
        
    return plot, p_t

reward_prob = [0.3, 0.7, 0.2]
num_arms = 3
horizon = 1000
learning_rate = np.sqrt(2*np.log(num_arms)/(horizon*num_arms))
plot, p_t = exp3(num_arms, horizon, learning_rate)
plt.figure(figsize=(10, 6))
for arm in range(num_arms):
    plt.plot(plot[:, arm], label=f'Arm {arm+1}')
plt.xlabel('Time step')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward of Each Arm Over Time')
plt.legend()
plt.show()
print(p_t)
