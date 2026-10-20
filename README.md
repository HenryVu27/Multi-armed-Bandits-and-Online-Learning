# Multi-armed bandits and online learning

This repo is from undergraduate research at the University of Alberta on sequential decision-making under uncertainty. The code runs classic multi-armed bandit algorithms on simulated arms and evaluates contextual bandits offline on logged news-recommendation data.

The research also covered Markovian and restless bandits. On the online-learning side, it covered the online primal-dual method and online algorithms that use machine-learned predictions. Those topics appear only in the slides and notes listed below. This repo has no code for them.

The blog post [Multi-Armed Bandits: Foundations](https://henryvu.blog/series/bandits/part1.html) explains the bandit setting and the upper confidence bound (UCB) algorithm in more depth.

## Contents

| Path | What it holds |
|---|---|
| `algorithms/Algorithms Comparison.py` | ε-greedy, UCB1 and explore-then-commit (ETC) on simulated Gaussian arms |
| `algorithms/Exp3.py` | Exp3 (exponential weights for exploration and exploitation) on simulated Bernoulli arms |
| `algorithms/Thompson.py` | Thompson sampling with Beta priors on simulated Bernoulli arms |
| `algorithms/Exp4.py` | Unfinished draft of Exp4 that still runs the Exp3 code |
| `project/MABs.ipynb` | Offline evaluation of ε-greedy, UCB and LinUCB (linear UCB) on logged news-recommendation data |
| `project/dataset.txt` | The 10,000 logged events the notebook uses |
| `visualization/` | Figures saved from `Algorithms Comparison.py` and `Exp3.py` |
| `slides/` | Seven slide decks on bandits, from the introduction to the Gittins and Whittle indices |
| `online_learning/` | Slide decks on online algorithms, from convex duality and the online primal-dual method to learning-augmented algorithms for online conversion |
| `Final Report.pdf` | Written notes with the derivations behind the slides, including regret bounds for ETC, UCB, Exp3 and Exp4 |

## Results

### Stochastic arms

`Algorithms Comparison.py` runs ε-greedy (ε = 0.1), UCB1 and ETC for 10,000 steps on seven arms with unit-variance Gaussian rewards and means between 1.0 and 5.0. ETC plays the arms in turn for the first 1,000 steps and then plays the arm with the highest sample mean.

<p>
<img src="visualization/eg%20vs%20ucb%20vs%20etc.png" alt="Cumulative regret of epsilon-greedy, UCB1 and explore-then-commit over 10,000 steps" width="49%">
<img src="visualization/eg%20vs%20ucb%20vs%20etc%20_n.png" alt="Average regret per step of epsilon-greedy, UCB1 and explore-then-commit" width="49%">
</p>

In this run UCB1 ends with the least regret. ETC collects most of its regret during the 1,000 exploration steps and then stays flat. ε-greedy never stops exploring, so its regret keeps growing linearly and its average regret per step (right) levels off above zero.

### Exp3

`Exp3.py` runs Exp3 for 1,500 rounds on five Bernoulli arms with success probabilities 0.1, 0.9, 0.2, 0.3 and 0.15. It uses the importance-weighted reward estimate and the learning rate $\sqrt{2\ln K/(nK)}$ for $K$ arms and horizon $n$ from Lattimore and Szepesvári. Exp3 is designed for rewards chosen by an adversary. This simulation draws them from fixed Bernoulli distributions.

<p>
<img src="visualization/exp3%20weights.png" alt="Probability Exp3 assigns to each of five arms over 1,500 rounds" width="49%">
<img src="visualization/exp3%20regret%20sublinear.png" alt="Exp3 average regret per step over 1,500 rounds" width="49%">
</p>

The probability on arm 2, the arm that pays off 90% of the time, rises to nearly 1 within about 600 rounds (left). Average regret per step falls from about 0.7 to about 0.05 by round 1,500 (right), which is consistent with sublinear regret.

The [estimated reward plot](visualization/exp3%20estimated%20reward.png) shows the cost of importance weighting. When Exp3 pulls an arm it rarely plays and gets no reward, that round's estimate for the arm is 1 - 1/p, where p is the arm's probability. For arm 3 this reached about -1,300. The `visualization/` folder also has the cumulative regret and each arm's cumulative estimated reward.

### Thompson sampling

`Thompson.py` runs Thompson sampling with Beta(1, 1) priors for 5,000 rounds on three Bernoulli arms with success probabilities 0.9, 0.5 and 0.2. It plots each arm's posterior mean and the per-step regret when run. Neither plot is saved in the repo.

### Contextual bandits on logged data

`project/MABs.ipynb` compares two policies that ignore context, ε-greedy and UCB, with LinUCB, which fits a ridge regression per arm on 10-feature context vectors. The data is 10,000 logged events in which a uniformly random policy showed one of 10 news articles and recorded a click (reward 1) or no click (reward 0).

Evaluation uses the replay method of Li et al. (2011). The policy sees each logged event in order. When it picks the article the log recorded, the event counts toward its reward and the policy updates. Otherwise the event is skipped. Each run stops after 800 matched events.

| Policy | Average reward over 800 matched events |
|---|---|
| ε-greedy, ε = 0.1 | 0.21 |
| UCB | 0.14 |
| LinUCB, α = 1 | 0.53 |
| LinUCB, α ≈ 0.026 | about 0.73 |

LinUCB, which uses the context, earns more than either policy that ignores it. Lowering its exploration parameter α from 1 raises its reward further. A sweep over 50 values of α between 0.001 and 1 picked α ≈ 0.026. That value was chosen on the same replayed events it is scored on, so its 0.73 is an optimistic estimate. The notebook sets no random seed, so the ε-greedy and UCB numbers shift a little between runs. The plots of the α sweep and of each policy's running average reward are in the notebook's saved outputs.

## Running the code

The scripts need NumPy and Matplotlib, and the notebook also needs Jupyter. There is no requirements file.

```bash
pip install numpy matplotlib jupyter
python "algorithms/Algorithms Comparison.py"
python algorithms/Exp3.py
python algorithms/Thompson.py
cd project && jupyter notebook MABs.ipynb
```

`Exp3.py` and `Thompson.py` fix the random seed at 32. `Algorithms Comparison.py` and the notebook do not. `Exp3.py` and `Thompson.py` create their last figures after the final `plt.show()` call, so run them with `%run` in Jupyter or IPython to see every figure. The notebook reads `dataset.txt` from its working directory, so start Jupyter from `project/`.

## References

- Tor Lattimore and Csaba Szepesvári, *Bandit Algorithms*, Cambridge University Press, 2020. The notes and the Exp3 implementation follow this book, and the Exp3 learning rate comes from its Theorem 11.2.
- Peter Auer, Nicolò Cesa-Bianchi and Paul Fischer, "Finite-time analysis of the multiarmed bandit problem," *Machine Learning*, 2002. Source of UCB1.
- Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund and Robert E. Schapire, "The nonstochastic multiarmed bandit problem," *SIAM Journal on Computing*, 2002. Source of Exp3.
- William R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, 1933. Source of Thompson sampling.
- Lihong Li, Wei Chu, John Langford and Robert E. Schapire, "A contextual-bandit approach to personalized news article recommendation," International Conference on World Wide Web, 2010, [arXiv:1003.0146](https://arxiv.org/abs/1003.0146). Source of LinUCB.
- Lihong Li, Wei Chu, John Langford and Xuanhui Wang, "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms," International Conference on Web Search and Data Mining, 2011, [arXiv:1003.5956](https://arxiv.org/abs/1003.5956). Source of the replay evaluation.
- Bo Sun, Russell Lee, Mohammad Hajiesmaili, Adam Wierman and Danny H. K. Tsang, "Pareto-optimal learning-augmented algorithms for online conversion problems," Advances in Neural Information Processing Systems, 2021, [arXiv:2109.01556](https://arxiv.org/abs/2109.01556). Presented in the `online_learning/` slides.
