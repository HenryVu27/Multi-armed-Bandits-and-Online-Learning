# Datasets

The dataset `dataset.txt` contains 10,000 instances corrresponding to distinct site visits by users-events in the language of this part. Each instance comprises 102 space-delimited columns of integers:
 - Column 1: The arm played by a uniformly-random policy out of 10 arms (news articles, numnbered 0-9)
 - Column 2: The reward received from the arm played|1 if the user clicked 0 otherwise; and
 - Columns 3-102: The 100-dim flattened context; 10 features per arm (incorporating the content of the article and its match with the visiting user)


## Implemented algorithms

1. ɛ-greedy 
2. Upper Confidence Bound (UCB)
3. LinUCB contextual MAB including evaluation and hyperparameter tuning [1]
Off-policy evaluation (OPE) [1] is used for evaluation.


## References 

[1] Lihong Li, Wei Chu, John Langford, Robert E. Schapire, ‘A Contextual-Bandit Approach to Personalized News Article Recommendation’, in Proceedings of the Nineteenth International Conference on World Wide Web (WWW’2010), Raleigh, NC, USA, 2010. 
https://arxiv.org/pdf/1003.0146.pdf