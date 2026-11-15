# Deep Hedging of an Equinox

Natixis × ENS [Challenge Data 2020](https://challengedata.ens.fr/challenges/38): we learn a weekly hedging strategy for a 5-year exotic option under transaction costs and a risk limit, using a neural network trained with the *deep hedging* framework.

Everything is in a single notebook, [`main.ipynb`](main.ipynb): data, P&L, model, training, evaluation and submission.

## The problem

We are **short** a 5-year *Equinox* (strike $K$ = barrier $B$ = $S_0$ = 1, reset at $R$ = 2.5 years, rebate $G$ = 0.05):

$$
\text{payoff} = \mathbf{1}_{\{S_R \ge B\}}\,G + \mathbf{1}_{\{S_R < B\}}\,(S_T-K)^+
$$

| rule | |
|---|---|
| Delta hedge | choose a position $\theta_t$ in the underlying, every week (260 weeks) |
| Vega hedge | imposed: buy 5Y calls at $t=0$ and at the reset $t=R$ |
| Risk limit | $\lvert\Delta_{\text{book}}(t) + \theta_t\rvert \le L$, for $L \in \{0.1, 0.2, 0.5\}$ |
| Costs | $10^{-4}$ on the underlying, 2% of the price on options |
| Score | entropic risk $\log \mathbb{E}\big[e^{-\text{PnL}}\big]$, averaged over the 3 limits (lower is better) |

The paths are simulated with stochastic volatility and spot–vol correlation, under the **historical measure** (drift $\mu = -3\%$/year, not arbitrage-free).

## Approach

**Deep hedging** (Buehler et al., 2019): parametrise the strategy with a network and minimise the risk of the final P&L directly over simulated paths.

$$
\theta_t = c_t + L\,\tanh\big(f(h_t)\big), \qquad h_t = \text{LSTM}(x_0, \dots, x_t)
$$

- $c_t$ is the centre of the allowed band, so the **risk limit holds by construction**.
- The LSTM only sees the past, so the strategy is **adapted** (no look-ahead).
- **Loss** = the challenge score, the entropic risk $\rho = \frac{1}{\lambda}\log \mathbb{E}\big[e^{-\lambda\,\text{PnL}}\big]$ with $\lambda = 1$.
- One model per limit $L$, trained on 25 000 supplementary paths and evaluated on the 5 000 official paths.

## Results

Entropic risk on the official validation file (5 000 paths):

| strategy | L = 0.1 | L = 0.2 | L = 0.5 | **average** |
|---|---|---|---|---|
| benchmark (hedge the book delta) | 0.0782 | 0.0782 | 0.0782 | 0.0782 |
| constant short tilt ($\theta_t = c_t - L$) | 0.0685 | 0.0612 | 0.0471 | **0.0589** |
| LSTM | 0.0685 | 0.0612 | 0.0471 | **0.0589** |

On the challenge platform, both the LSTM and the constant strategy scored **0.0597**.

**What we learned**
- The network converges to the **constant short tilt**: always sit at the short edge of the allowed band. That beats the benchmark by about 25%.
- **Why it works:** the spot has a negative drift, so a short position earns money on average. With exponential utility the optimal tilt is $\mu / (\lambda\sigma^2) \approx -0.75$, outside every band, so the limit binds.
- **Why nothing beats it:** the main risk is the jump at the reset (rebate vs call) and the imposed vega hedge, and neither can be offset within $\lvert r \rvert \le L$. Adding volatility as an input or reacting near the reset did not improve on the constant strategy in our tests.

## Repository

```
.
├── main.ipynb     # the whole project
└── README.md
```

## How to run

**1. Get the data.** It is available on the [challenge page](https://challengedata.ens.fr/challenges/38) after registering, and is not redistributed here. Put these files next to `main.ipynb`:

| expected file name | content |
|---|---|
| `Supplementary Training Data (1).csv` | 25 000 training paths |
| `Challenge Data Validation.csv` | the 5 000 official paths (X) |
| `Challenge Data Validation (1).csv` | the submission template (Y format) |

**2. Install the dependencies** (tested with Python 3.10):

```bash
pip install numpy pandas torch matplotlib jupyter
```

**3. Run `main.ipynb` from top to bottom.** Training runs on CPU. The last cell writes `submission_lstm_entropic.csv` in the platform format.

## Notes

- **Scoring function.** The organisers did not publish their scoring code. Section 6 of the notebook uses a scoring function found in a participant's notebook ([Abdulaziz1607/NATIXIS-CHALLENGE](https://github.com/Abdulaziz1607/NATIXIS-CHALLENGE)). Our P&L reproduces it to about $10^{-6}$. It is consistent with the platform: 0.0589 locally vs 0.0597 on the platform, a gap compatible with the platform scoring a subset of the paths.
- **Two conventions follow that scoring function** rather than the challenge text. The score is the entropic risk, not the "average replication cost" described on the page. The risk limit is centred on $-(\Delta^E_t - \psi_t\Delta^C_t)$.

## References

- H. Buehler, L. Gonon, J. Teichmann, B. Wood, *Deep Hedging*, Quantitative Finance, 2019. [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)
- O. Croissant, *Deep Hedging of an Equinox*, Natixis, Challenge Data 2020 (presentation slides).
- [Challenge page](https://challengedata.ens.fr/challenges/38) on Challenge Data (ENS).
