from sprinkler import *

np.random.seed(0)

data = np.load('data.npz')

blue_x = data['blue_x']
blue_z = data['blue_z']
green_y = data['green_y']
green_z = data['green_z']

target_drop_ratio = blue_x.shape[0]/green_y.shape[0]
tol_drop_ratio = 0.1

N_drops = 10000
N_expts = 2000

accepted = None

for i in range(N_expts):

    x = np.random.uniform(8)
    y = np.random.uniform(8)

    blue_x, blue_z, green_y, green_z = sprinkler(x, y, N_drops)
    drop_ratio = blue_x.shape[0]/green_y.shape[0]

    # Test ABC criteria
    if np.abs(target_drop_ratio - drop_ratio) < tol_drop_ratio:
        if accepted is None:
            accepted = np.array([x ,y])
        else:
            accepted = np.vstack((accepted, np.array([x, y])))
plt.scatter(accepted[:, 0], accepted[:, 1], color='none', edgecolor='black')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('abc_criteria1.png')

