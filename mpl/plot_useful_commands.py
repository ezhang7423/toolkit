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
plt.axvline(mean, color="red", label=f"Mean: ${mean:.2f}$")
plt.axvline(
    mean + std,
    color="red",
    linestyle="--",
    alpha=0.8,
    label=f"Mean + $\sigma$: ${mean+std:.2f}$",
)
plt.axvline(
    mean - std,
    color="red",
    linestyle="--",
    alpha=0.8,
    label=f"Mean - $\sigma$: ${mean-std:.2f}$",
)
plt.legend()

# * add horizontal line
plt.axhline(
    0.2,
    color="black",
    linestyle="--",
    alpha=0.5,
    label="y=0.2",
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


# * Be sure to only pick integer tick locations.
from matplotlib import ticker

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# * Plot colorbar

import matplotlib.cm as cm
from matplotlib.colors import Normalize

color_mapper = sns.color_palette("flare", as_cmap=True)  # goes from 0 to 1
norm = Normalize(0, 10)
plot(
    plt.scatter,
    x,
    y,
    cmap=color_mapper,  #!
    c=y,  #!
    norm=norm,  #!
    xlabel="PC2 (Generosity)",
    ylabel="Happiness",
    title="Analysis of PC 2 effect on Happiness, 2023",
)
cbar = plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap=color_mapper), label="Add label here"  #!
)


# * Plot linear regression line

from scipy.stats import linregress

m, b, *_ = linregress(x, y)
plt.axline(
    xy1=(0, b),
    slope=m,
    label=f"Regression Line : $y = {m:.3f}x {b:+.1f}$",
    color="black",
    linestyle="--",
)
plt.legend()


# * plot mean and std of timings across different widths

widths = [2, 3, 5]
timings = np.random.rand(3, 5)
plot(plt.plot, np.mean(timings, axis=1))
plot(
    plt.fill_between,
    range(len(widths)),
    np.mean(timings, axis=1) - np.std(timings, axis=1),
    np.mean(timings, axis=1) + np.std(timings, axis=1),
    alpha=0.2,
    title="Time taken to train varying model widths over 5 seeds",
    xlabel="Width",
    ylabel="Avg Time (s) $\pm \sigma$",
)
# * show only the widths actually tested on the x ticks
plt.xticks(range(len(widths)), widths)
plt.show()

