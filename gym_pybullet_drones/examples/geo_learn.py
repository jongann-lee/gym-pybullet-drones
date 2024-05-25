"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

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

class CustomNetwork(torch.nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 300,
        last_layer_dim_vf: int = 300,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Actor network (actions)
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 400), torch.nn.ReLU(), torch.nn.Linear(400,300), torch.nn.Tanh()
        )
        # Critic network (Q-function)
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 400), torch.nn.ReLU(), torch.nn.Linear(400,300), torch.nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

def run( output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # train_env = GeoHoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    train_env = make_vec_env(GeoHoverAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, train=True),
                             n_envs=1,
                             seed=0
                             )
    train_norm_env = VecNormalize(train_env, norm_obs= True, norm_reward= True)
    eval_env = make_vec_env(GeoHoverAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                             n_envs=1,
                             seed=0
                             )
    eval_norm_env = VecNormalize(eval_env, norm_obs= True, norm_reward= False, training=False)

    # train_norm_env = VecNormalize.load(output_folder + "/save-05.24.2024_22.23.19/norm_param.pkl",train_norm_env)
    # eval_norm_env = VecNormalize.load(output_folder + "/save-05.24.2024_22.23.19/norm_param.pkl",eval_norm_env)
    # eval_norm_env.training = False
    # eval_norm_env.norm_reward = False

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           #net_arch=[128, 128]
                           net_arch=dict(vf=[256, 256], pi=[256, 128])
                           )
    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=dict(pi=[400, 300], qf=[400, 300])
                            )

    n_actions = train_env.action_space.shape[-1]
    actionnoise = NormalActionNoise(mean=np.zeros(n_actions), sigma = 0.1*np.ones(n_actions))

    model = TD3('MlpPolicy',
                train_norm_env,
                policy_kwargs = offpolicy_kwargs,
                buffer_size= 200000,
                learning_starts= 10000,
                learning_rate = 1e-3, 
                action_noise=actionnoise,
                batch_size = 256,
                gradient_steps= 1,
                train_freq = 1,
                gamma = 0.98,
                tau = 0.001,
                # tensorboard_log=filename+'/tb/',
                verbose=1)
    
    #### Optional: load a previous model
    #load_name = os.path.join(output_folder, 'save-05.24.2024_22.23.19/best_model.zip') # save-05.06.2024_02.09.37
    #model = TD3.load(load_name)
    #model.set_parameters(load_path_or_dict=load_name)
    #model.set_env(train_env)
    #model.action_noise = actionnoise
    #model.learning_rate = 0.00005

    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 
    else:
        target_reward = 2000. 
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    
    eval_callback = EvalCallback(eval_norm_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(5e5) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)


    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)
    train_norm_env.save(filename + '/norm_param.pkl')

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = TD3.load(path)

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

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
