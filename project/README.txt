# Datasets

The dataset dataset.txt contains 10,000 instances of users and news articles. Each instance contains 102 columns with the following information:
 - Column 1: The arm played by a uniformly random policy (arms numbered 0-9)
 - Column 2: The reward received from the arm played|1 if the user clicked 0 otherwise; and
 - Columns 3-102: The 100-dim flattened context; 10 features per arm (incorporating the content of the article and its match with the visiting user)


# Implemented algorithms

1. ɛ-greedy 
2. Upper Confidence Bound (UCB)
3. LinUCB contextual MAB including evaluation and hyperparameter tuning [1]
Off-policy evaluation (OPE) [1] is used for evaluation.


# References 

[1] Li, Lihong, et al. "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms." Proceedings of the fourth ACM international conference on Web search and data mining. 2011.
https://arxiv.org/pdf/1003.0146.pdf
