import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

data = pd.read_csv('galaxies.csv')
columns = ['redshift', 'stellar_mass', 'deVRad_r', 'deVAB_r', 'expRad_r', 'expAB_r']

X = data[['redshift', 'stellar_mass']].to_numpy()
mu = X.mean(axis=0)
sigma = X.std(axis=0)
X = (X-mu)/sigma
#print(X)
pca = PCA(n_components=1)
pca.fit(X)
W = pca.components_
T = pca.transform(X)
plt.scatter(T, np.zeros(len(T)), s=5, facecolor='None', edgecolor='tab:blue')
plt.plot([-5,6], [0,0], color='red')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('reduce.png')
