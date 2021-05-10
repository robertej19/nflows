import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
bar_0_10 = ax.bar(np.arange(0,10), np.arange(1,11), color="k")
bar_10_100 = ax.bar(np.arange(0,10), np.arange(30,40), bottom=np.arange(1,11), color="g")
# create blank rectangle
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
ax.legend([extra, bar_0_10, bar_10_100], ("My explanatory text", "0-10", "10-100"))
plt.show()

