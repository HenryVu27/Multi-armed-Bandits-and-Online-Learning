import matplotlib.pyplot as plt
import numpy as np

class Exp3:
    def __init__(self, num_arms, learning_rate):
        self.num_arms = num_arms
        self.learning_rate = learning_rate
        self.weights = np.ones(num_arms)
        self.probabilities = np.ones(num_arms) / num_arms
        self.rewards = np.zeros(num_arms)  # Store cumulative rewards
        self.history = []  # Store history of chosen arms and rewards

    def select_arm(self):
        """Select an arm based on the computed probabilities."""
        chosen_arm = np.random.choice(self.num_arms, p=self.probabilities)
        return chosen_arm

    def update(self, chosen_arm, reward):
        """Update the weights and probabilities."""
        # Compute the estimated reward
        x_hat = reward / self.probabilities[chosen_arm]
        
        # Update the rewards for the chosen arm
        self.rewards[chosen_arm] += reward
        
        # Update weight for the chosen arm
        self.weights[chosen_arm] *= np.exp(self.learning_rate * x_hat / self.num_arms)
        
        # Update the probabilities for all arms
        self.probabilities = self.weights / np.sum(self.weights)

    def run(self, horizon):
        """Run the EXP3 algorithm for a specified horizon and store the history."""
        for t in range(horizon):
            chosen_arm = self.select_arm()
            # In a real-world scenario, here we would get a reward for the chosen arm.
            # For simulation, we will use a random reward.
            reward = np.random.rand()  # This should be replaced with the actual reward
            self.update(chosen_arm, reward)
            self.history.append((chosen_arm, reward))
        return self.rewards

# Initialize the EXP3 algorithm with 5 arms and a learning rate of 0.1
exp3 = Exp3(num_arms=5, learning_rate=0.1)

# Run the EXP3 algorithm for a horizon of 1000 time steps
horizon = 1000
rewards = exp3.run(horizon)

# Plotting the cumulative reward of each arm over time
cumulative_rewards = np.zeros((horizon, exp3.num_arms))
for t, (arm, reward) in enumerate(exp3.history):
    cumulative_rewards[t] = cumulative_rewards[t-1]  # start with the last state
    cumulative_rewards[t, arm] += reward  # add reward to the selected arm

# Plot the results
plt.figure(figsize=(10, 6))
for arm in range(exp3.num_arms):
    plt.plot(cumulative_rewards[:, arm], label=f'Arm {arm+1}')
plt.xlabel('Time step')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward of Each Arm Over Time')
plt.legend()
plt.show()
