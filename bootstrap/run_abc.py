from sprinkler import *

np.random.seed(0)

data = np.load('data.npz')

blue_x = data['blue_x']
blue_z = data['blue_z']
green_y = data['green_y']
green_z = data['green_z']

target_drop_ratio = blue_x.shape[0]/green_y.shape[0]
target_mean_drop_height = np.concatenate((blue_z, green_z)).mean()

# Generate histograms of the true data
bins = 50
target_blue_heatmap, blue_x_edges, blue_y_edges = np.histogram2d(blue_x, blue_z, bins = bins)
target_blue_heatmap /= len(blue_x)

target_green_heatmap, green_x_edges, green_y_edges = np.histogram2d(green_y, green_z, bins=bins)
target_green_heatmap /= len(green_y)

tol_drop_ratio = 0.1
tol_mean_drop_height = 0.02
tol_heatmap_abs_err = 2

N_drops = 10000
N_expts = 2000

accepted = None

for i in range(N_expts):

    x = np.random.uniform(8)
    y = np.random.uniform(8)

    blue_x, blue_z, green_y, green_z = sprinkler(x, y, N_drops)
    drop_ratio = blue_x.shape[0]/green_y.shape[0]
    mean_drop_height = np.concatenate((blue_z, green_z)).mean()
    blue_heatmap = np.histogram2d(blue_x, blue_z, bins = bins)[0]/len(blue_x)
    green_heatmap = np.histogram2d(green_y, green_z, bins = bins)[0]/len(green_y)

    # Test ABC criteria
    criteria1 = np.abs(target_drop_ratio - drop_ratio) < tol_drop_ratio
    criteria2 = np.abs(target_mean_drop_height - mean_drop_height) < tol_mean_drop_height
    criteria3 = (np.abs(target_blue_heatmap - blue_heatmap).sum() + np.abs(target_green_heatmap - green_heatmap).sum()) < tol_heatmap_abs_err

    if criteria1 and criteria2 and criteria3:
        if accepted is None:
            accepted = np.array([x ,y])
        else:
            accepted = np.vstack((accepted, np.array([x, y])))
plt.scatter(accepted[:, 0], accepted[:, 1], color='none', edgecolor='black')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('abc_criteria3.png')
