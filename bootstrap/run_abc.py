from sprinkler import *

np.random.seed(0)

data = np.load('data.npz')

blue_x = data['blue_x']
blue_z = data['blue_z']
green_y = data['green_y']
green_z = data['green_z']

target_drop_ratio = blue_x.shape[0]/green_y.shape[0]
target_mean_drop_height = np.concatenate((blue_z, green_z)).mean()

tol_drop_ratio = 0.1
tol_mean_drop_height = 0.02

N_drops = 10000
N_expts = 2000

accepted = None

for i in range(N_expts):

    x = np.random.uniform(8)
    y = np.random.uniform(8)

    blue_x, blue_z, green_y, green_z = sprinkler(x, y, N_drops)
    drop_ratio = blue_x.shape[0]/green_y.shape[0]
    mean_drop_height = np.concatenate((blue_z, green_z)).mean()

    # Test ABC criteria
    criteria1 = np.abs(target_drop_ratio - drop_ratio) < tol_drop_ratio
    criteria2 = np.abs(target_mean_drop_height - mean_drop_height) < tol_mean_drop_height
    if criteria1 and criteria2:
        if accepted is None:
            accepted = np.array([x ,y])
        else:
            accepted = np.vstack((accepted, np.array([x, y])))
plt.scatter(accepted[:, 0], accepted[:, 1], color='none', edgecolor='black')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('abc_criteria2.png')
