"""
This is a script designed to test the learned geometric tuner models
"""

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.GeoHoverAviary import GeoHoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel
from gym_pybullet_drones.control.GeometricControl import GeometricControl

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('geo') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_DRONEMODEL = DroneModel("cf2x")
DEFAULT_MA = False

gui = DEFAULT_GUI
output_folder=DEFAULT_OUTPUT_FOLDER
record_video = DEFAULT_RECORD_VIDEO
colab = DEFAULT_COLAB
plot = True

filename = os.path.join(output_folder, 'save-05.23.2024_10.31.15')

if os.path.isfile(filename+'/best_model.zip'):
    path = filename+'/best_model.zip'
    path_2 = filename + '/final_model.zip'
else:
    print("[ERROR]: no model under the specified path", filename)
model = TD3.load(path_2)

#### Show (and record a video of) the model's performance ##
test_vec_env = make_vec_env(GeoHoverAviary,
                             env_kwargs=dict(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record = record_video),
                             n_envs=1,
                             seed=0
                             )
test_vec_env = VecNormalize.load(filename + "/norm_param.pkl", test_vec_env)
test_vec_env.training = False
test_vec_env.norm_reward = False
test_env = test_vec_env.envs[0]
test_env_nogui = make_vec_env(GeoHoverAviary,
                            env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                            n_envs=1,
                            seed=0
                            )
test_env_nogui = VecNormalize.load(filename + "/norm_param.pkl", test_env_nogui)
test_env_nogui.training = False
test_env_nogui.norm_reward = False

logger = Logger(logging_freq_hz=int(test_env.PYB_FREQ),
            num_drones=1,
            output_folder=output_folder,
            colab=colab
            )

mean_reward, std_reward = evaluate_policy(model,
                                            test_env_nogui,
                                            n_eval_episodes=10
                                            )
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

obs = test_vec_env.reset()
start = time.time()
for i in range(test_env.PYB_STEPS_PER_CTRL):
    obs2 = test_env.observation_buffer[i][:]
    #print("Obs:", obs2)
    logger.log(drone=0,
                timestamp=i/test_env.PYB_FREQ + 0.1,
                state=obs2,
                control=np.zeros(12)
                )
#print(obs)
for i in range((test_env.EPISODE_LEN_SEC)*test_env.PYB_FREQ):
    if (i % test_env.PYB_STEPS_PER_CTRL) == 0:
        action, _states = model.predict(obs,
                                    deterministic=True
                                    )
        obs, reward, dones, info = test_vec_env.step(action)
        #if truncated == True: print("\t Truncated")
        act2 = action.squeeze()
        print("\tAction", action)
        
    
    obs2 = test_env.observation_buffer[i % test_env.PYB_STEPS_PER_CTRL][:]
    #print("Obs:", obs2)
    
    
    #print("Obs:", obs2, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
    
    if DEFAULT_OBS == ObservationType.KIN:
        logger.log(drone=0,
            timestamp=i/test_env.PYB_FREQ + 0.1,
            state=obs2,
            control=np.zeros(12)
            )

    #test_env.render()
    #print(terminated)
    sync(i, start, test_env.PYB_TIMESTEP)
    if dones:
        obs = test_vec_env.reset()
test_env.close()

if plot and DEFAULT_OBS == ObservationType.KIN:
    logger.plot()