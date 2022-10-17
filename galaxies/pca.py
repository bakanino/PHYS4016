from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('galaxies.csv')
columns = ['redshift', 'stellar_mass', 'deVRad_r', 'deVAB_r', 'expRad_r', 'expAB_r']

X = data[columns].to_numpy()
mu = X.mean(axis=0)
sigma = X.std(axis=0)

X = (X-mu)/sigma

pca = PCA(n_components=3)
pca.fit(X)
T = pca.transform(X)

mu = T.mean(axis=0)
sigma = T.std(axis=0)

T = (T-mu)/sigma

spirals = T[data['class_label'] == 'spiral']
ellipticals = T[data['class_label'] == 'elliptical']

# Visualise your data
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(ellipticals[:, 0], ellipticals[:, 1], ellipticals[:, 2], s = 50, marker = 'o', alpha = 0.5, label = 'Elliptical', color = 'tab:blue', edgecolor = 'tab:blue')
ax.scatter(spirals[:, 0], spirals[:, 1], spirals[:, 2], s = 50, marker = 'o', label = 'Spiral', alpha = 0.2, color = 'tab:orange', edgecolor = 'tab:orange')

# Project onto the z plane
ax.scatter(ellipticals[:,0], ellipticals[:, 1], marker = '.', zdir = 'z', zs = -10, alpha = 0.05, color = 'tab:blue')
ax.scatter(spirals[:,0], spirals[:, 1], marker = '.', zdir = 'z', zs = -10, alpha = 0.05, color = 'tab:orange')

# Project onto the x plane
ax.scatter(ellipticals[:,1], ellipticals[:, 2], marker = '.', zdir = 'x', zs = -10, alpha = 0.05, color = 'tab:blue')
ax.scatter(spirals[:,1], spirals[:, 2], marker = '.', zdir = 'x', zs = -10, alpha = 0.05, color = 'tab:orange')

# Project onto the y plane
ax.scatter(ellipticals[:,0], ellipticals[:, 2], marker = '.', zdir = 'y', zs = 10, alpha = 0.05, color = 'tab:blue')
ax.scatter(spirals[:,0], spirals[:, 2], marker = '.', zdir = 'y', zs = 10, alpha = 0.05, color = 'tab:orange')

plt.legend()
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([-10, 10])
plt.savefig('pca.png')
