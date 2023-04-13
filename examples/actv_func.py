
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-5, 5+1e-5, 1e-5)

relu = np.maximum(x, 0)

sigmoid = 1 / (1 + np.exp(-x))

tanH = np.tanh(x)

plt.figure()
plt.axvline(0, linestyle="dashed", color="black")
plt.axhline(0, linestyle="dashed", color="black")
plt.plot(x, relu, color="red", label="ReLU", linewidth=3)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.legend(frameon=False)
plt.title(r"$ReLU(x): max(0, x)$", fontsize=18)

plt.figure()
plt.axvline(0, linestyle="dashed", color="black")
plt.axhline(0, linestyle="dashed", color="black")
plt.plot(x, sigmoid, color="green", label="Sigmoide", linewidth=3)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.legend(frameon=False)
plt.title(r"$\sigma (x) = \frac{1}{(1 - e^{-x})}$", fontsize=18)

plt.figure()
plt.axvline(0, linestyle="dashed", color="black")
plt.axhline(0, linestyle="dashed", color="black")
plt.plot(x, tanH, color="blue", label="TanH", linewidth=3)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks(np.arange(-1, 1.5, 0.5))
plt.legend(frameon=False)
plt.title(r"$\sigma(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$", fontsize=18)

plt.show()
