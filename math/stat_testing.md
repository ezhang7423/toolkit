# Python Statistical Tests Cheatsheet

## 1. Student's t-test

**When to use:** 
- Compare means of two groups
- Continuous dependent variable, categorical independent variable with 2 levels
- Assumes normal distribution and equal variances

**Test name:** t-test

**Example:**
```python
import numpy as np
from scipy import stats

# Independent two-sample t-test
group1 = np.array([5.2, 4.8, 5.5, 4.9, 5.1])
group2 = np.array([5.8, 6.1, 5.9, 6.2, 5.7])
t_statistic, p_value = stats.ttest_ind(group1, group2)
print(f"Independent t-test: t={t_statistic:.4f}, p={p_value:.4f}")

# Paired t-test
before = np.array([180, 190, 175, 185, 195])
after = np.array([175, 185, 170, 180, 190])
t_statistic, p_value = stats.ttest_rel(before, after)
print(f"Paired t-test: t={t_statistic:.4f}, p={p_value:.4f}")

# One-sample t-test
sample = np.array([25.5, 26.2, 24.8, 25.1, 25.9])
popmean = 25
t_statistic, p_value = stats.ttest_1samp(sample, popmean)
print(f"One-sample t-test: t={t_statistic:.4f}, p={p_value:.4f}")
```

## 2. ANOVA (Analysis of Variance)

**When to use:** 
- Compare means of three or more groups
- Continuous dependent variable, categorical independent variable with 3+ levels
- Assumes normal distribution and equal variances

**Test name:** F-test

**Example:**
```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Create a sample dataset
data = {
    'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'value': [4.2, 3.8, 4.1, 5.7, 5.9, 5.8, 7.2, 6.8, 7.1]
}
df = pd.DataFrame(data)

# Perform one-way ANOVA
model = ols('value ~ C(group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results:")
print(anova_table)
```

## 3. Chi-Square Test of Independence

**When to use:** 
- Test relationship between two categorical variables
- Assumes expected frequencies are at least 5 in each cell

**Test name:** Chi-square test

**Example:**
```python
import numpy as np
from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = np.array([
    [30, 20, 10],
    [15, 25, 20]
])

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square test: chi2={chi2:.4f}, p={p_value:.4f}, dof={dof}")
```

## 4. Pearson Correlation

**When to use:** 
- Measure strength and direction of linear relationship between two continuous variables
- Assumes bivariate normal distribution

**Test name:** Pearson's r

**Example:**
```python
import numpy as np
from scipy import stats

x = np.array([1.2, 2.4, 3.6, 4.8, 6.0])
y = np.array([2.1, 3.5, 4.8, 6.2, 7.5])

r, p_value = stats.pearsonr(x, y)
print(f"Pearson correlation: r={r:.4f}, p={p_value:.4f}")
```

## 5. Spearman Rank Correlation

**When to use:** 
- Measure strength and direction of monotonic relationship between two variables
- Use when data is ordinal or not normally distributed

**Test name:** Spearman's rho

**Example:**
```python
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 4, 6])

rho, p_value = stats.spearmanr(x, y)
print(f"Spearman correlation: rho={rho:.4f}, p={p_value:.4f}")
```

## 6. Mann-Whitney U Test

**When to use:** 
- Non-parametric alternative to t-test
- Compare two independent samples
- No assumption of normal distribution

**Test name:** Mann-Whitney U test

**Example:**
```python
import numpy as np
from scipy import stats

group1 = np.array([72, 68, 75, 71, 70])
group2 = np.array([80, 78, 82, 79, 81])

statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
print(f"Mann-Whitney U test: U={statistic:.4f}, p={p_value:.4f}")
```

## 7. Wilcoxon Signed-Rank Test

**When to use:** 
- Non-parametric alternative to paired t-test
- Compare two related samples
- No assumption of normal distribution

**Test name:** Wilcoxon signed-rank test

**Example:**
```python
import numpy as np
from scipy import stats

before = np.array([180, 190, 175, 185, 195])
after = np.array([175, 185, 170, 180, 190])

statistic, p_value = stats.wilcoxon(before, after)
print(f"Wilcoxon signed-rank test: W={statistic:.4f}, p={p_value:.4f}")
```

## 8. Kruskal-Wallis H-test

**When to use:** 
- Non-parametric alternative to one-way ANOVA
- Compare three or more independent samples
- No assumption of normal distribution

**Test name:** Kruskal-Wallis H-test

**Example:**
```python
import numpy as np
from scipy import stats

group1 = np.array([72, 68, 75, 71, 70])
group2 = np.array([80, 78, 82, 79, 81])
group3 = np.array([85, 87, 83, 86, 84])

h_statistic, p_value = stats.kruskal(group1, group2, group3)
print(f"Kruskal-Wallis H-test: H={h_statistic:.4f}, p={p_value:.4f}")
```

## 9. Shapiro-Wilk Test

**When to use:** 
- Test if a sample comes from a normally distributed population
- Recommended for small sample sizes (n < 50)

**Test name:** Shapiro-Wilk test

**Example:**
```python
import numpy as np
from scipy import stats

data = np.array([3.2, 3.5, 3.1, 3.4, 3.3, 3.6, 3.2, 3.5])

statistic, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk test: W={statistic:.4f}, p={p_value:.4f}")
```

## 10. Levene's Test

**When to use:** 
- Test for homogeneity of variances across groups
- Often used before performing t-test or ANOVA

**Test name:** Levene's test

**Example:**
```python
import numpy as np
from scipy import stats

group1 = np.array([72, 68, 75, 71, 70])
group2 = np.array([80, 78, 82, 79, 81])
group3 = np.array([85, 87, 83, 86, 84])

statistic, p_value = stats.levene(group1, group2, group3)
print(f"Levene's test: W={statistic:.4f}, p={p_value:.4f}")
```

Remember to always check the assumptions of each test and interpret the results in the context of your research question and data characteristics.
