# Python Statistical Tests Cheatsheet

## 1. Student's t-test

**When to use:** 
- Compare means of two groups
- Continuous dependent variable, categorical independent variable with 2 levels
- Assumes normal distribution and equal variances

**Test name:** t-test

**Example:**
```python
from scipy import stats

# Independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(group1, group2)

# Paired t-test
t_statistic, p_value = stats.ttest_rel(before, after)

# One-sample t-test
t_statistic, p_value = stats.ttest_1samp(sample, popmean)
```

## 2. ANOVA (Analysis of Variance)

**When to use:** 
- Compare means of three or more groups
- Continuous dependent variable, categorical independent variable with 3+ levels
- Assumes normal distribution and equal variances

**Test name:** F-test

**Example:**
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Perform one-way ANOVA
model = ols('value ~ C(group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

## 3. Chi-Square Test of Independence

**When to use:** 
- Test relationship between two categorical variables
- Assumes expected frequencies are at least 5 in each cell

**Test name:** Chi-square test

**Example:**
```python
from scipy.stats import chi2_contingency

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```

## 4. Pearson Correlation

**When to use:** 
- Measure strength and direction of linear relationship between two continuous variables
- Assumes bivariate normal distribution

**Test name:** Pearson's r

**Example:**
```python
from scipy import stats

r, p_value = stats.pearsonr(x, y)
```

## 5. Spearman Rank Correlation

**When to use:** 
- Measure strength and direction of monotonic relationship between two variables
- Use when data is ordinal or not normally distributed

**Test name:** Spearman's rho

**Example:**
```python
from scipy import stats

rho, p_value = stats.spearmanr(x, y)
```

## 6. Mann-Whitney U Test

**When to use:** 
- Non-parametric alternative to t-test
- Compare two independent samples
- No assumption of normal distribution

**Test name:** Mann-Whitney U test

**Example:**
```python
from scipy import stats

statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
```

## 7. Wilcoxon Signed-Rank Test

**When to use:** 
- Non-parametric alternative to paired t-test
- Compare two related samples
- No assumption of normal distribution

**Test name:** Wilcoxon signed-rank test

**Example:**
```python
from scipy import stats

statistic, p_value = stats.wilcoxon(before, after)
```

## 8. Kruskal-Wallis H-test

**When to use:** 
- Non-parametric alternative to one-way ANOVA
- Compare three or more independent samples
- No assumption of normal distribution

**Test name:** Kruskal-Wallis H-test

**Example:**
```python
from scipy import stats

h_statistic, p_value = stats.kruskal(group1, group2, group3)
```

## 9. Shapiro-Wilk Test

**When to use:** 
- Test if a sample comes from a normally distributed population
- Recommended for small sample sizes (n < 50)

**Test name:** Shapiro-Wilk test

**Example:**
```python
from scipy import stats

statistic, p_value = stats.shapiro(data)
```

## 10. Levene's Test

**When to use:** 
- Test for homogeneity of variances across groups
- Often used before performing t-test or ANOVA

**Test name:** Levene's test

**Example:**
```python
from scipy import stats

statistic, p_value = stats.levene(group1, group2, group3)
```

Remember to always check the assumptions of each test and interpret the results in the context of your research question and data characteristics.
