import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
plt.scatter(X[:,0], X[:,1], s=5, alpha=0.2)
plt.quiver([0,0], [0,0], W[:,0], W[:,1], scale_units='xy', scale=1/pca.explained_variance_)
plt.axis('equal')
plt.xlabel('Redshift')
plt.ylabel('Stellar mass')
plt.savefig('pca.png')
