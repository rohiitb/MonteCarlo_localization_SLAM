import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class map_t:
    """
    This will maintain the occupancy grid and log_odds. 
    """

    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szy = int(np.ceil((s.ymax - s.ymin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

        s.free_prob_thresh = 0.2
        s.log_odds_free = np.log(s.free_prob_thresh / (1 - s.free_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). 
        """

        x_clipped = np.clip(x, s.xmin, s.xmax)
        y_clipped = np.clip(y, s.ymin, s.ymax)
        x_grid = np.ceil((x_clipped - s.xmin) / s.resolution).astype(np.int16)  # - 1
        y_grid = np.ceil((y_clipped - s.ymin) / s.resolution).astype(np.int16)  # - 1
        cells = np.vstack((x_grid, y_grid))  # (2, len(x))

        return cells


class slam_t:

    def __init__(s, resolution=0.05, Q=1e-3 * np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = 1e-11 * np.eye(3)
        # s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

        s.curr_part = None    #keeping the track of current particle
        s.robot_trajectory = None     #keeping track of robot trajectory
        s.odom = None     #keeping track of robot odometry
        s.unk_cells = None 

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d' % (split, split, idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d' % (split, split, idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t'] - t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135, 135 + s.lidar_angular_resolution,
                                   s.lidar_angular_resolution) * np.pi / 180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1 / 9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n) / float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """

        n = p.shape[1]
        r = np.random.uniform(0, 1 / n)

        c = w[0]
        i = 0
        w_new = deepcopy(w)
        pose_new = deepcopy(p)
        for m in range(n):
            u = r + (m - 1) / n

            while u > c:
                i += 1
                c += w[i]

            w_new[m] = w_new[i]
            pose_new[:, m] = pose_new[:, i]

        return pose_new, w_new

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w - w.max()).sum())

    def rays2world(s, p, d, imu_data, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """


        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data

        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)

        # 1. from lidar distances to points in the LiDAR frame
        x_lidar = d * np.cos(angles)
        y_lidar = d * np.sin(angles)
        z_lidar = np.zeros(len(d))
        ones = np.ones(len(d))
        lid_data = np.vstack((x_lidar, y_lidar, z_lidar, ones))

        # 2. from LiDAR frame to the body frame
        t_head_lidar = np.array([[0, 0, s.lidar_height]])
        t_body_head = np.array([0, 0, 0.33])
        H_lidar2head = euler_to_se3(0, 0, 0, t_head_lidar)
        H_head2body = euler_to_se3(0, head_angle, neck_angle, t_body_head)

        # 3. from body frame to world frame
        t_body_disp = np.array([p[0], p[1], 0.93])
        H_body2world = euler_to_se3(imu_data[0], imu_data[1], imu_data[2], t_body_disp)

        H_lidar2world = H_body2world @ H_head2body @ H_lidar2head
        final_cood = H_lidar2world @ lid_data

        final_cood_clipped = final_cood[:2, final_cood[2] > 0.1]

        return final_cood_clipped

    def unoccupied_cells_to_obstacles(s, cur_loc, obs_loc):
        free_cells = cur_loc
        x_s = cur_loc[0]
        y_s = cur_loc[1]

        for i in obs_loc.T:
            x_o = i[0]
            y_o = i[1]
            dir = i - cur_loc.T
            dist = int(np.linalg.norm(dir))
            x = np.linspace(x_s, x_o, dist, endpoint=False, dtype=int)
            y = np.linspace(y_s, y_o, dist, endpoint=False, dtype=int)
            new_free_cells = np.vstack((x.T, y.T))
            print("new_free_cells: ", new_free_cells.shape)
            free_cells = np.hstack((free_cells, new_free_cells))
            print("free_cells: ", free_cells.shape)
        free_cells = np.unique(free_cells, return_index=False, axis=1)

        return free_cells

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)


        pose_cur = np.array([s.lidar[t]['xyth'][0], s.lidar[t]['xyth'][1], s.lidar[t]['rpy'][2]])
        pose_prev = np.array([s.lidar[t - 1]['xyth'][0], s.lidar[t - 1]['xyth'][1], s.lidar[t - 1]['rpy'][2]])
        return smart_minus_2d(pose_cur, pose_prev)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """

        delta_pose = s.get_control(t)
        for i in range(s.p.shape[1]):
            noise = np.random.multivariate_normal([0, 0, 0], s.Q)
            s.p[:, i] = smart_plus_2d(s.p[:, i], delta_pose)
            s.p[:, i] = smart_plus_2d(s.p[:, i], noise)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        pi = np.log(w) + obs_logp
        qi = slam_t.log_sum_exp(pi)
        ri = np.exp(pi - qi)
        return ri

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """

        imu_data = s.lidar[t]['rpy']

        t_head_ang = s.joint['head_angles'][1][s.find_joint_t_idx_from_lidar(t)]  # head angle at t
        t_neck_ang = s.joint['head_angles'][0][s.find_joint_t_idx_from_lidar(t)]  # neck angle at t

        odo = (s.lidar[t]['xyth'][:2])
        grid_odom = s.map.grid_cell_from_xy(odo[0], odo[1])
        grid_odom = np.array([[grid_odom[0]], [grid_odom[1]]])
        if s.odom is None:
            s.odom = grid_odom
        else:
            s.odom = np.hstack((s.odom, grid_odom))

        if t == 0:
            likely_grid_pt = s.map.grid_cell_from_xy(*s.p.T[0, :2])  # current grid point position
            lidar_scanned_points = s.rays2world(s.p.T[0], s.lidar[t]['scan'], imu_data, t_head_ang, t_neck_ang,
                                                s.lidar_angles)  # obstacles detected from the likely grid point
            occ_cells = s.map.grid_cell_from_xy(lidar_scanned_points[0], lidar_scanned_points[1])  # converting to cell
            s.map.cells[occ_cells[0], occ_cells[1]] = 1  # giving high probability to the cells
            s.curr_part = s.p.T[0]  # for plotting     #add to object
            free_cells = s.unoccupied_cells_to_obstacles(likely_grid_pt, occ_cells)  # unoccupied cells between obstacle and current location
            unk_cells = np.ones_like(s.map.cells, dtype=bool)
            unk_cells[free_cells[0], free_cells[1]] = 0
            unk_cells[occ_cells[0], occ_cells[1]] = 1
            s.unk_cells = np.argwhere(unk_cells)          #add to object

            s.map.cells = np.zeros_like(s.map.cells)
            s.map.cells[occ_cells[0], occ_cells[1]] = 1
            s.map.cells[free_cells[0], free_cells[1]] = 0

            s.robot_trajectory = s.map.grid_cell_from_xy(s.curr_part[0], s.curr_part[1])    #add to object
            s.robot_trajectory = np.reshape(s.robot_trajectory, (2, 1))

        else:
            log_obs = np.zeros(s.p.shape[1])
            for i in range(s.p.shape[1]):
                lidar_scanned_points = s.rays2world(s.p.T[i], s.lidar[t]['scan'], imu_data, t_head_ang, t_neck_ang, s.lidar_angles)
                occ = s.map.grid_cell_from_xy(lidar_scanned_points[0], lidar_scanned_points[1])
                log_obs[i] = np.sum(s.map.cells[occ[0], occ[1]])

            s.w = s.update_weights(s.w, log_obs)
            likely_part_id = np.argmax(s.w)
            likely_part = s.p.T[likely_part_id]

            likely_grid_pt = s.map.grid_cell_from_xy(likely_part[0], likely_part[1])

            s.curr_part = likely_part                         #add to object ...for plotting
            s.robot_trajectory = np.hstack((s.robot_trajectory, likely_grid_pt.reshape((2, 1))))

            lidar_scanned_points = s.rays2world(likely_part, s.lidar[t]['scan'], imu_data, t_head_ang, t_neck_ang, s.lidar_angles)
            likely_occ = s.map.grid_cell_from_xy(lidar_scanned_points[0], lidar_scanned_points[1])
            print("occ: ", occ.shape)
            likely_free = s.unoccupied_cells_to_obstacles(occ, likely_grid_pt)

            for i in range(likely_occ.shape[1]):          #updating the log odds
                s.map.log_odds[likely_occ[0, i], likely_occ[1, i]] += s.lidar_log_odds_occ

            s.map.log_odds[likely_free[0], likely_free[1]] += s.lidar_log_odds_free             #updating the free cells
            s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)

            occupied_cells = s.map.log_odds >= s.map.log_odds_thresh
            free_cells = s.map.log_odds <= s.map.log_odds_free

            s.unk_cells = np.logical_and(np.logical_not(free_cells), np.logical_not(occupied_cells))
            s.map.cells = np.zeros_like(s.map.cells)
            s.map.cells[occupied_cells] = 1           #assigning final probabilities
            s.map.cells[free_cells] = 0

            s.resample_particles()



    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1 / np.sum(s.w ** 2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e / s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')

