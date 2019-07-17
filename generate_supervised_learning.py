from metric_learn import NCA, MMC
import matplotlib.pyplot as plt
import numpy as np


X = np.array([[-3, 2],
              [-2.9, -0.86],
              [4.24, 3.08],
              [4.08, -0.2],
              [1.58, -3.14],
              [-0.38, -3.76]])

y = np.array([0, 2, 1, 1, 0, 2])

def plot_points(X, y, name):
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()

  x_center = (x_min + x_max)/2
  y_center = (y_min + y_max)/2

  max_diff = max(x_max - x_min, y_max - y_min)
  margin = max_diff / 20

  y_lims = (y_center - max_diff/2 - margin, y_center + max_diff/2 + margin)
  x_lims = (x_center - max_diff/2 - margin, x_center + max_diff/2 + margin)



  plt.figure()

  plt.scatter(X[:, 0], X[:, 1], c=y)
  plt.xlim(*x_lims)
  plt.ylim(*y_lims)
  plt.axis('equal')
  plt.savefig(name)


plot_points(X, y, 'supervised_without_metric')

nca = NCA()
nca.fit(X, y)

X_e = nca.transform(X)

plot_points(X_e, y, 'supervised_with_metric')
