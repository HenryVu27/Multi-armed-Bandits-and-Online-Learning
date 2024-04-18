# Overview
This project aims to compare the performance of different bandit algorithms for Yahoo's news recommendations dataset. The replay method by Li et al. [1] is implemented for 
off-policy evaluation.
During evaluation, replay takes in the new policy (to be evaluated) and the logged policy events. If the new policy chooses the same action as the logged policy, 
the event is added to the history and the reward is updated. If not, the event is ignored with no reward update.

# Datasets
The dataset dataset.txt contains 10,000 instances of users and news articles (logged events). Each instance contains 102 columns with the following information:
 - Column 1: The arm played by a uniformly random policy (arms numbered 0-9).
 - Column 2: The reward received from the arm played (1 if clicked otherwise 0).
 - Columns 3-102: The 100-dim flattened context; 10 features each arm for 10 arms.


# Implemented algorithms
1. ɛ-greedy 
2. Upper Confidence Bound (UCB) with rho = 1
3. LinUCB contextual MAB including hyperparameter tuning [1]
Off-policy evaluation (OPE) [1] is used for evaluation.


# References 
[1] Li, Lihong, et al. "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms." Proceedings of the fourth ACM international conference on Web search and data mining. 2011. https://arxiv.org/pdf/1003.0146.pdf
