from sprinkler import *
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(0)

x = 2.40
y = 1.19
blue_x, blue_z, green_y, green_z = sprinkler(x, y, 100000)

ax = visualise_droplets(blue_x, blue_z, green_y, green_z)
ax.scatter(x, y, 0, color = 'none', edgecolor = 'black', alpha = 1)
plt.savefig('model.png')

print('{:.2f}, {:.2f}'.format(x, y))
