import numpy as np
import matplotlib.pyplot as plt

def sprinkler(x, y, ndrops):
    '''
    x and y are the location of the sprinkler.
    ndrops is the number of water drops.
    '''

    # Plant sprinkler on the lawn
    spigot_xy = np.array([x, y])
    sp_slope = spigot_xy[1]/spigot_xy[0]
    spigot_rt = [np.sqrt(spigot_xy[0]**2+spigot_xy[1]**2),np.arctan(spigot_xy[1]/spigot_xy[0])]

    # generate random angles for drops
    da_nw = (np.random.uniform(size=ndrops)) * np.pi / 2.0
    da_nw = -da_nw + spigot_rt[1]
    da_ew = (np.random.uniform(size=ndrops)) * np.pi / 2.0
    da_ew = da_ew + spigot_rt[1]
    slopes_nw = np.tan(da_nw)
    slopes_ew = np.tan(da_ew)

    # cut out all droplets that are moving in the wrong quadrant
    slopes_nw = slopes_nw.compress((slopes_nw > 0).flat)
    slopes_ew = slopes_ew.compress((slopes_ew > 0).flat)

    # find intercepts on wall
    # first the North wall is at coordinate [ * ,-1 * spigot_xy[1]]
    nw_drops_x = -spigot_xy[1]/slopes_nw
    ew_drops_y = -spigot_xy[0]*slopes_ew

    # Distance from spigot to the intercept with the wall
    nw_drops_dist=np.sqrt(nw_drops_x**2 + spigot_xy[1]**2)
    ew_drops_dist=np.sqrt(spigot_xy[0]**2 + ew_drops_y**2)

    # Now add the velocity of the droplet (distribution)
    v_mu, v_sigma = 10.0, 0.5  # mean droplet velocity and standard deviation
    ndrops_nw, ndrops_ew = nw_drops_dist.shape[0],ew_drops_dist.shape[0]
    nw_drops_v = np.random.normal(loc = v_mu, scale = v_sigma, size = ndrops_nw)
    ew_drops_v = np.random.normal(loc = v_mu, scale = v_sigma, size = ndrops_ew)

    # all drops launched at 45 degrees angle
    drop_theta = np.pi*45./180
    drv = np.cos(drop_theta) #for 45 degrees cos = sin = 1/sqrt(2)

    # include the vertical z-direction component of the droplets
    nw_drops_tof = nw_drops_dist / (nw_drops_v * drv) # time-of-flight for drops to hit wall
    ew_drops_tof = ew_drops_dist / (ew_drops_v * drv)
    gravity = -9.8        # meters / sec2
    nw_drops_z = nw_drops_v * drv * nw_drops_tof + 0.5 * gravity * nw_drops_tof**2
    ew_drops_z = ew_drops_v * drv * ew_drops_tof + 0.5 * gravity * ew_drops_tof**2

    #cull drops below ground level and re-center corner of wall to origin
    nwix = (nw_drops_z > 0)
    nwz = nw_drops_z.compress(nwix.flat)
    nw_drops_x += spigot_xy[0]            # move origin to corner of wall
    nwx = nw_drops_x.compress(nwix.flat)  # cull below-ground drops
    ewix = (ew_drops_z > 0)
    ewz = ew_drops_z.compress(ewix.flat)
    ew_drops_y += spigot_xy[1]            # move origin to corner of wall
    ewy = ew_drops_y.compress(ewix.flat)  # cull below-ground drops

    return nwx, nwz, ewy, ewz # blue_x, blue_z, green_y, green_z

def visualise_droplets(blue_x, blue_z, green_y, green_z):

    blue_drops = blue_x.shape[0]
    green_drops = green_y.shape[0]

    fig = plt.figure(figsize = (15, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(blue_x, np.zeros(blue_drops), blue_z, s = 2, color = 'tab:blue', alpha = 0.5)
    ax.scatter(np.zeros(green_drops), green_y, green_z, s = 2, color = 'tab:green', alpha = 0.5)

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.axes.set_xlim(8, -12); ax.axes.set_ylim(8, -12); ax.axes.set_zlim(0,4)

    # Draw the wall
    plt.plot([-12, 0], [0, 0], [0, 0], color = 'black')
    plt.plot([0, 0], [-12, 0], [0, 0], color = 'black')
    plt.plot([0, 0], [0, 0], [0, 4], color = 'black')
    plt.plot([0, 0], [-12, -12], [0, 4], color = 'black')
    plt.plot([-12, -12], [0, 0], [0, 4], color = 'black')
    plt.plot([-12, 0], [0, 0], [4, 4], color = 'black')
    plt.plot([0, 0], [-12, 0], [4, 4], color = 'black')

    ax.view_init(elev = 20, azim = 210)
    plt.tight_layout()

    return ax
