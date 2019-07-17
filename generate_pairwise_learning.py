from metric_learn import ITML, MMC
import matplotlib.pyplot as plt
import numpy as np


X = np.array([[-3, 2],
              [-2.9, -0.86],
              [4.24, 3.08],
              [4.08, -0.2],
              [1.58, -3.14],
              [-0.38, -3.76]])

c = np.array([[0, 4], [1, 2], [3, 5], [2, 3]])

pairs = X[c]
y_pairs = np.array([1, -1, -1, 1])

def plot_points(pairs, y_pairs, name):
  x_min, x_max = pairs[:, :, 0].min(), pairs[:, :, 0].max()
  y_min, y_max = pairs[:, :, 1].min(), pairs[:, :, 1].max()

  x_center = (x_min + x_max)/2
  y_center = (y_min + y_max)/2

  max_diff = max(x_max - x_min, y_max - y_min)
  margin = max_diff / 20

  y_lims = (y_center - max_diff/2 - margin, y_center + max_diff/2 + margin)
  x_lims = (x_center - max_diff/2 - margin, x_center + max_diff/2 + margin)



  plt.figure()

  for i, edge in enumerate(pairs):
      plt.plot(edge[:, 0], edge[:, 1], c='green' if y_pairs[i] == 1 else
      'red', alpha=0.3)
  plt.scatter(pairs[:, 0, 0], pairs[:, 0, 1], c='b')
  plt.scatter(pairs[:, 1, 0], pairs[:, 1, 1], c='b')
  plt.xlim(*x_lims)
  plt.ylim(*y_lims)
  plt.axis('equal')
  plt.savefig(name)


plot_points(pairs, y_pairs, 'pairs_without_metric')

mmc = ITML()
mmc.fit(pairs, y_pairs)

X_e = mmc.transform(X)
pairs = X_e[c].copy()

plot_points(pairs, y_pairs, 'pairs_with_metric')
