from sprinkler import *
from mpl_toolkits.mplot3d import Axes3D

data = np.load('data.npz')

blue_x = data['blue_x']
blue_z = data['blue_z']
green_y = data['green_y']
green_z = data['green_z']

visualise_droplets(blue_x, blue_z, green_y, green_z)
plt.savefig('sprinkler.png')

target_drop_ratio = blue_x.shape[0]/green_y.shape[0]
print('Drop ratio: {:.4f}'.format(target_drop_ratio))
