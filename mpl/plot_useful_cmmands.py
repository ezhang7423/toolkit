from matplotlib import pyplot as plt
import numpy as np


# * limit plot window
plt.ylim(-3, 3)
plt.xlim(-3, 3)

# * rotate xlabels
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment="right")


# * errorbars
x = [f"bar {i}" for i in range(32)]
y = np.linspace(1, 0.2, 32)
yerr = np.full((32,), 0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".", color="Black")

# * change figure size
plt.figure(figsize=(5.8, 3))

# * add vertical line
x = np.linspace(1, 0.2, 32)
mean = np.mean(x)
std = np.std(x)
plt.axvline(mean, color="red", label="Mean")
plt.axvline(
    mean + std,
    color="red",
    linestyle="--",
    alpha=0.8,
    label="Mean + $1\sigma$",
)

# * correlation matrix
import seaborn as sns

cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.gca().tick_params(
    top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True
)
plt.minorticks_off()


# * set y scale to log
plt.yscale("log")

# * set y scale to symmetric log (allow negative values)
plt.yscale("symlog")
