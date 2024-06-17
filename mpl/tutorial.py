############################################
# Preamble

import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')

############################################
# science lineplot

plt.plot([0, 1, 2], [1, 2, 3])
plt.plot([0, 1, 2], [0, 1, 3])
plt.title('the  quick fox jumped over the lazy dog 2 $x = 2$')
plt.legend(['line 1', 'line 2'])
plt.xlabel('x-axis $\mu$')
plt.ylabel("y-axis $\psi$")
plt.show()

############################################
# minimal quibbler

from pyquibbler import initialize_quibbler, iquib

initialize_quibbler()
import matplotlib.pyplot as plt

x = iquib(0.5)
y = 1 - x
plt.plot([0, 1], [1, 0], "-")
plt.plot([0, x, x], [y, y, 0], "--", marker="D")
plt.title(x, fontsize=20)


############################################
