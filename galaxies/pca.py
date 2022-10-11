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
pca = PCA()
pca.fit(X)
W = pca.components_
T = pca.transform(X)
plt.scatter(T[:,0], T[:,1], s=5, alpha=0.2)
plt.plot([-5,6], [0,0], color='red')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('projection.png')
