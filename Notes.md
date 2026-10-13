# Risk Measures

A risk measure is a mathematical function that assigns a numerical value to the risk of a portfolio.

**Simply:** it is a tool that quantifies financial risk, such as potential losses.

## 1. Value at Risk (VaR)

The Value at Risk at confidence level $\gamma$ is defined as

$$
\text{VaR}_{\gamma}(X)
= \inf y\in\mathbb{R}:F_X(y)\geq\gamma,
$$

where:

- $X$ is the random variable representing the portfolio loss at a fixed time horizon $T$.
- $F_X(y)=P(X\leq y)$ is the cumulative distribution function (CDF).
- $\gamma$ is the confidence level, such as $0.99$.

**Simply:** VaR gives a loss threshold that is not exceeded with approximately $\gamma$ probability under the model.

### 1.1. Simple example

Suppose we have 100 possible simulated losses. After sorting them from smallest to largest, the loss around the 99th percentile is $0.12$. Then $\text{VaR}_{0.99}(X)\approx 0.12.$

We interpret this as:

- Approximately 99% of modeled outcomes have losses at or below the VaR threshold.
- Approximately 1% of outcomes lie above the threshold.

Questions that can be drawn from this example:

1. How do we model losses?
2. Why do we sort the simulated losses?
3. How do we find the $n$-th percentile?

### 1.2. What is an empirical quantile, and why do we use it?

In practice, we may not know the exact distribution of the loss $X$. Instead, we simulate $M$ losses:

$$
X^{(1)},X^{(2)},\ldots,X^{(M)}.
$$

These are $M$ simulated outcomes of the same random variable $X$, usually obtained from independent Monte Carlo simulations.

Sort them in increasing order:

$$
X_{(1)}\leq X_{(2)}\leq\cdots\leq X_{(M)}.
$$

The notation $X_{(k)}$ means the $k$-th smallest simulated loss.

An empirical estimate of VaR is

$$
\widehat{\text{VaR}}_{\gamma}(X)
= X_{(\lceil M\gamma\rceil)},
$$

where $\lceil M\gamma\rceil$ is the smallest integer greater than or equal to $M\gamma$. This is one common empirical quantile convention; software packages may use other interpolation conventions.

For example, if $M=10{,}000$ and $\gamma=0.99$, then

$$
\lceil M\gamma\rceil
=\lceil 10{,}000\times0.99\rceil
=9{,}900.
$$

So, with this convention, we take the 9,900th smallest simulated loss as the empirical 99% quantile.

**Why do we use this formula?**

The definition of VaR asks for the loss threshold whose cumulative probability reaches the confidence level $\gamma$. Sorting the simulated losses approximates their distribution, and the corresponding order statistic estimates that threshold.

In Python, using zero-based indexing:

```python
X_sort = np.sort(X)
index = int(np.ceil(gamma * M)) - 1
VaR = X_sort[index]
```

The formula `int(gamma * M) - 1` used in the original C++ task selects the 9,900th observation when $M=10{,}000$ and $\gamma=0.99$, because Python indices start at zero. For these values, it agrees with the order-statistic convention above.

### 1.3. What is a stochastic process?

A stochastic process is a collection of random variables indexed by time:

$$
\{X_t:t\geq 0\}.
$$

At each fixed time $t$, $X_t$ is a random variable. A simulated path is one realization of the process over time.

For example, one simulated path over five time points might be

$$
X_0=0,\quad X_{t_1}=0.03,\quad
X_{t_2}=0.08,\quad X_{t_3}=0.04,\quad X_T=0.07.
$$

Another simulation gives a different path.

Therefore:

- **A random variable at a fixed time:** $X_T$, the loss at the end of the horizon.
- **A stochastic process:** $\{X_t:0\leq t\leq T\}$, the losses over the whole horizon.
- **A simulated path:** one realization of the process across the simulated time points.
- **Monte Carlo simulations:** many simulated paths, each providing a terminal loss and, when needed, a maximum loss.

When computing terminal VaR, we estimate the distribution of $X_T$ using the terminal values from many simulated paths. We do not work with only one path.

## 2. Intra-horizon Value at Risk (iVaR)

### 2.1. Why do we need iVaR?

Suppose a portfolio begins at a loss level of zero. During the next 10 working days, its cumulative loss may rise to $0.18$, then fall to $0.07$ by the end of the period.

The final loss is $0.07$, which misses the fact that the loss reached $0.18$ during the period.

So we need to keep track of the losses throughout the horizon.

### 2.2. What is iVaR?

Let $X_t$ represent the cumulative loss at time $t$, with $X_0=0$.

Define the running maximum over the horizon by

$$
\overline{X}_T=\sup_{0\leq t\leq T}X_t.
$$

Here, $\overline{X}_T$ is the maximum loss reached along one path during the interval $[0,T]$. It is also called the supremum of the process over the horizon.

The intra-horizon Value at Risk is

$$
\text{iVaR}_{\gamma}(X_T)
=\text{VaR}_{\gamma}(\overline{X}_T).
$$

**Simply:** we compute the VaR of the distribution of maximum losses rather than the distribution of terminal losses.

Since

$$
\overline{X}_T\geq X_T
$$

for each path, the corresponding quantiles satisfy

$$
\text{iVaR}_{\gamma}(X_T)
\geq \text{VaR}_{\gamma}(X_T),
$$

provided both quantiles use the same confidence level and consistent quantile definitions.

**Note:**

- Here, we work over a time interval.
- We work with many simulated loss paths.
- For each path, we find its maximum loss.
- We then compute the VaR of those maximum losses.

### 2.3. Example: VaR versus iVaR

Imagine that we simulate five paths over 10 working days.

| Simulated path | Loss at the end $X_T$ | Maximum loss during the period $\overline{X}_T$ |
|---:|---:|---:|
| 1 | 0.02 | 0.04 |
| 2 | 0.05 | 0.11 |
| 3 | 0.01 | 0.15 |
| 4 | 0.08 | 0.19 |
| 5 | 0.04 | 0.23 |

For each path, the maximum loss is at least as large as the terminal loss. To estimate VaR, we take a quantile of the five terminal losses. To estimate iVaR, we take the same quantile of the five maximum losses.

With only five paths, the 99th percentile is not estimated reliably; this example illustrates the difference between the two quantities rather than providing a useful 99% risk estimate.

### 2.4. VaR versus iVaR

| Feature | VaR | iVaR |
|---|---|---|
| What is measured? | Loss at the end of the horizon | Maximum loss during the horizon |
| Random variable | $X_T$ | $\overline{X}_T$ |
| Formula | $\text{VaR}_{\gamma}(X_T)$ | $\text{VaR}_{\gamma}(\overline{X}_T)$ |
| Requires path simulation? | Terminal simulations are enough | Usually requires path simulation |
| Main question | How large could the final loss be at a given confidence level? | How large could the loss get at any point during the period? |

### 2.5. Why do we take the supremum?

The supremum is taken **along each individual path**, over time. It is not the maximum across all simulated paths.

For example, suppose one path has losses

$$
0,\ 0.03,\ 0.12,\ 0.07,\ 0.09.
$$

Its maximum loss is

$$
\overline{X}_T=\max(0,0.03,0.12,0.07,0.09)=0.12.
$$

We repeat this operation for every simulated path. If we simulate $M$ paths, we obtain

$$
\overline{X}_T^{(1)},\overline{X}_T^{(2)},\ldots,
\overline{X}_T^{(M)}.
$$

We then sort these $M$ maximum losses and take their empirical $\gamma$-quantile. That gives the Monte Carlo estimate of iVaR.

For a continuous-time process, the supremum is taken over every time in $[0,T]$. In a numerical simulation, we usually approximate it using the simulated time points. If jumps occur, we should also update the running maximum after each jump.

## 3. How to compute VaR and iVaR

Two main approaches are:

1. **By distribution:** derive or obtain the distribution, calculate its CDF, and find the required quantile.
2. **By Monte Carlo simulation:** simulate many outcomes or paths, sort the relevant values, and estimate the quantile empirically.

### 3.1. VaR from a known distribution

Suppose the loss $X$ has a known CDF $F_X$. We find the value $q$ satisfying

$$
F_X(q)=\gamma,
$$

when the CDF is continuous and the quantile is uniquely determined. Equivalently,

$$
q=F_X^{-1}(\gamma).
$$

For example, if $X\sim\mathcal{N}(\mu,s^2)$, then

$$
\text{VaR}_{\gamma}(X)
=\mu+s\Phi^{-1}(\gamma),
$$

where $\Phi$ is the standard normal CDF.

### 3.2. VaR by Monte Carlo

1. Simulate $M$ terminal losses $X_T^{(1)},\ldots,X_T^{(M)}$.
2. Sort them in increasing order.
3. Select the observation corresponding to the confidence level $\gamma$.

### 3.3. iVaR by Monte Carlo

1. Simulate $M$ paths over the horizon.
2. For each path, calculate the maximum loss $\overline{X}_T^{(k)}$.
3. Sort the $M$ maximum losses.
4. Select the empirical $\gamma$-quantile.

## 4. Loss-level hitting risk

The task also asks for the probability that the loss reaches or exceeds a specified threshold $h$ during the time horizon.

The hitting event is

$$
\sup_{0\leq t\leq T}X_t\geq h.
$$

The hitting risk is

$$
w=P(\sup_{0\leq t\leq T}X_t\geq h).
$$

Using $M$ simulated paths, its Monte Carlo estimate is

$$
\widehat{w}
=\frac{1}{M}\sum_{k=1}^{M}
\mathbf{1}_{\{\overline{X}_T^{(k)}\geq h\}},
$$

where the indicator is 1 if path $k$ reaches or exceeds $h$, and 0 otherwise.

### What are we given, and what is asked?

In the Kou model task, we are given:

- Drift: $\gamma_0=0.2$.
- Volatility: $\sigma=0.3$.
- Jump intensity: $\lambda=3$.
- Positive-jump probability: $P=0.6$.
- Positive and negative exponential rates: $\Lambda_+=15$, $\Lambda_-=10$.
- Loss threshold: $h=0.15$.
- Time horizon: $T=10/252$ years.
- Number of simulations: $M=10{,}000$.

We are asked to estimate the probability that the cumulative loss reaches or exceeds $0.15$ during the 10-working-day horizon.

### Are we using VaR or iVaR to compute hitting risk?

The hitting risk is **not itself VaR or iVaR**. It is a probability of a threshold-crossing event. However, it is closely related to iVaR because both use the maximum loss along each path.

For a fixed threshold $h$,

$$
w=P(\overline{X}_T\geq h).
$$

This is the upper-tail probability of the maximum-loss random variable. iVaR, by contrast, is a quantile of the distribution of that same maximum-loss variable.

Therefore:

- **Hitting risk:** given $h$, estimate the probability that the maximum loss reaches or exceeds $h$.
- **iVaR:** given a confidence level $\gamma$, estimate the corresponding quantile of the maximum-loss distribution.
- **VaR:** estimate the quantile of the terminal-loss distribution.

With a continuous distribution, if $\text{iVaR}_{\gamma}(X_T)=q$, then approximately $\gamma$ of the maximum losses are at or below $q$, and $1-\gamma$ are above $q$. The hitting risk at threshold $h$ instead measures the probability that the maximum loss is at least $h$.
