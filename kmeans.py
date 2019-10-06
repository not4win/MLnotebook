from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('file.csv')
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2)
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
k = 2
C_x = np.random.randint(0, np.max(X)-20, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)))
print("Initial Centroids")
print(C)

plt.scatter(f1, f2)
plt.scatter(C_x, C_y, marker='*', s=200, c='Black')

C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
error = dist(C, C_old, None)

while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
print("Centroid values")
print(C) 
