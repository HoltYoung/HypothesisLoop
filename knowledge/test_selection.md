# Choosing a statistical test

This is the lookup table for the hypothesis loop. Pick the test that matches
the **shape** of the question, the **type** of the variables, and the
**distribution** of the data. When in doubt, prefer the non-parametric or
robust variant — it costs you a bit of power but rarely lies to you.

## Comparing two means

- **t-test (Welch's, two-sample)** — continuous outcome, two groups, roughly
  normal within each group. Welch's variant does not assume equal variances;
  prefer it as the default. Effect size: Cohen's d (small=0.2, medium=0.5,
  large=0.8).
- **Mann-Whitney U** — same setup, but the data are skewed, ordinal, or have
  outliers heavy enough to break t. Tests whether one distribution
  stochastically dominates the other; not a test of medians unless shapes
  match. Effect size: rank-biserial r.
- **Paired-sample variants** — when each observation in group A is matched to
  one in group B (before/after on the same subject, twin pairs, etc.), use
  the **paired t-test** or **Wilcoxon signed-rank**. Don't use a two-sample
  test on paired data — you'll badly understate power.

## Comparing more than two groups

- **One-way ANOVA** — continuous outcome, ≥3 groups, roughly normal within
  each, similar variances (Levene's test < 0.05 is a yellow flag). Effect
  size: η² (eta-squared) or ω². If the F-test rejects, follow up with
  **Tukey HSD** for all-pairs comparisons.
- **Kruskal-Wallis** — non-parametric counterpart. Same setup, skewed or
  ordinal data. Follow-up: pairwise Mann-Whitney with Bonferroni correction.

## Categorical associations

- **Chi-squared (χ²) test of independence** — two categorical variables,
  contingency table, expected count in every cell ≥5. Effect size: Cramér's V.
- **Fisher's exact test** — same question, but at least one cell has expected
  count <5 (small samples, rare categories). Default to Fisher's whenever a
  2×2 table has any zeros or near-zeros.

## Correlations

- **Pearson r** — both variables continuous, linear relationship, joint
  distribution roughly bivariate-normal, no extreme outliers. Reports linear
  association strength in [-1, 1].
- **Spearman ρ** — when the relationship is monotonic but not necessarily
  linear, or one variable is ordinal, or there are outliers. Operates on
  ranks; robust by construction.
- **Kendall τ** — alternative to Spearman, more conservative on small
  samples. Use when the sample is small (n<30) and tie-handling matters.

## Regression

- **Linear regression (OLS)** — continuous outcome, linear relationship,
  approximately normal residuals, homoscedasticity, no severe
  multicollinearity (check **VIF** — values >10 are bad, >5 worth a look).
  Effect sizes worth reporting: **R²** (overall), **adjusted R²** (penalizes
  extra predictors), and standardized coefficients (β) for each predictor.
- **Logistic regression** — binary outcome (0/1, yes/no, ≤50K vs >50K). Same
  multicollinearity concerns. Report **odds ratios with 95% confidence
  intervals**, not just p-values. An OR of 1.5 means the odds increase 50%
  per unit of the predictor — but always pair with the CI; an OR of 1.5 with
  CI [0.4, 5.6] is statistical noise.
- **Multinomial / ordinal logistic** — outcome with >2 unordered or ordered
  categories. More fragile; use only when the binary collapse loses the
  question.
- **Pre-flight checks for any regression**: residual plot (look for funnel /
  curvature), VIF, Cook's distance for outliers. Don't skip these.

## Distribution shape

- **Shapiro-Wilk** — tests normality; works for **small to moderate n**
  (n ≤ ~5000). Most powerful normality test for moderate samples.
- **Kolmogorov-Smirnov (KS)** — tests against any reference distribution
  (not just normal). Less powerful than Shapiro for normality specifically.
- **Anderson-Darling** — like KS, but more sensitive to the tails.

> **Caveat that earns its keep:** with **n > ~5000** every real-world dataset
> will reject normality at α=0.05 because the test detects trivial
> deviations. At that point, **visual inspection beats the test**: plot a
> histogram with a normal overlay, or a Q-Q plot. If the deviation isn't
> visible to the eye, it doesn't matter for practical inference.

## Effect size beats p-value

A p-value tells you whether an effect is **detectable**, not whether it is
**big enough to matter**. With n=30,000 (typical for the Adult census),
nearly any non-zero correlation will hit p<0.001 and nearly every group
difference will be "significant." That doesn't mean it's real, important, or
worth talking about.

Always report at least one effect size next to a p-value:

- **Cohen's d** for mean differences (small=0.2, medium=0.5, large=0.8).
- **η² / ω²** for ANOVA (small=0.01, medium=0.06, large=0.14).
- **r / R²** for correlations and regression fit (R²<0.05 = trivial,
  0.05–0.25 = modest, 0.25–0.5 = substantial, >0.5 = strong).
- **Odds ratios with 95% CI** for logistic regression. An OR whose CI crosses
  1.0 is not a finding regardless of the headline p-value.
- **Cramér's V** for chi-squared (small=0.1, medium=0.3, large=0.5).

If a hypothesis confirms with p<0.001 but a Cohen's d of 0.05, the honest
report is: **statistically detectable, practically negligible.** Don't
overstate the result.

## Library implementations — do not roll your own

For anything beyond a basic t-test / chi-squared / correlation, use the canonical
library. Hand-rolled statistics are almost always wrong.

- **Likelihood-ratio test** (nested logistic regressions): use `statsmodels`,
  NOT your own formula.

  ```python
  import statsmodels.api as sm
  from scipy import stats
  full = sm.GLM(y, sm.add_constant(X_full), family=sm.families.Binomial()).fit()
  reduced = sm.GLM(y, sm.add_constant(X_reduced), family=sm.families.Binomial()).fit()
  lr_stat = 2 * (full.llf - reduced.llf)
  df_diff = full.df_model - reduced.df_model
  p_value = stats.chi2.sf(lr_stat, df_diff)
  ```

  An LRT is **NOT** the sum of squared differences between predicted
  probabilities. That is mathematically meaningless.

- **ANOVA (multi-way):** `statsmodels.formula.api.ols` + `statsmodels.stats.anova_lm`.
- **Logistic regression:** `statsmodels.api.Logit` or `sklearn.linear_model.LogisticRegression`.
- **GLM beyond logistic:** `statsmodels.api.GLM` with the appropriate family
  (Poisson, Gamma, Negative Binomial).
- **Mixed-effects:** `statsmodels.formula.api.mixedlm`.
- **Bootstrap CIs:** `scipy.stats.bootstrap`.
- **Mediation/moderation:** `statsmodels` or `pingouin`.

If you're tempted to implement a statistic from a textbook formula, **check the
library first.** Correctness > performance in this loop.

## Quick decision recipe

1. What is the **outcome variable**? Continuous → t/ANOVA/regression
   families. Categorical → χ²/Fisher/logistic. Ordinal → Spearman/Kruskal.
2. How many **groups or predictors** are involved?
3. Are the parametric assumptions plausible (normality, equal variance, no
   crippling outliers)? If not, fall back to the non-parametric variant or a
   rank-based test.
4. Always pair the test with an **effect size** and, when relevant, a
   **confidence interval**. P-values alone are not enough.
