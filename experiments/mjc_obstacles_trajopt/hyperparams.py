""" Hyperparameters for MJC 2D navigation policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.cost.cost_fk import CostFK
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY, RAMP_QUADRATIC
from gps.utility.mjc_xml_extract import *
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_obstacles_trajopt/'

CONDITIONS = 1

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    'filename': [EXP_DIR + 'gen_model_'+str(i)+'.xml' for i in range(0, 0+CONDITIONS)],
    # convention : first indices are training, last are testing
    'train_conditions': [i for i in range(CONDITIONS)],
    'test_conditions': [],
    'iterations': 50,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    # 'filename': ['./mjc_models/particle2d_obstacle_11.xml', './mjc_models/particle2d_obstacle_11.xml'],
    # 'x0': [np.array([0., -0.3, 0., 0.]), np.array([0., 0., 0., 0.])],
    'filename': common['filename'],
    'x0': [extract_x0(f) for f in common['filename']],
    'target_state': [extract_target(f) for f in common['filename']],
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
    'meta_include': [],
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'iterations': common['iterations'],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_var': 10.0,
    'stiffness': 0.01,
    'stiffness_vel': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['cost'] = {
    'type': CostFK,
    'ramp_option': RAMP_QUADRATIC,
    'target_end_effector': np.array([0.43, -0.75, 0.0]),
    'wp': np.array([1, 1, 1]),
    'l1': 1.0,
    'l2': 1.0,
    'alpha': 1e-5,
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 5,
        'min_samples_per_cluster': 40,
        'max_samples': 40,
        'strength': 1,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['plot_controller_dist'] = False
common['plot_dynamics_prior'] = True
common['info'] = generate_experiment_info(config)

