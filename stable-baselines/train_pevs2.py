#%%
import time
import numpy as np
import matplotlib.pyplot as plt

import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.evaluation import evaluate_policy
#%%

from pev_battery_charge.envs.PEVBatteryChargeCentral import PEVBatteryChargeCentral
from pev_battery_charge.envs.config_pev import get_config
config = get_config(notebook=True)
config.num_agents = 4
config.n_pevs = 6

#%%
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PEVBatteryChargeCentral(config)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
#%%
from stable_baselines3.common.cmd_util import make_vec_env

#%%

ALGO = A2C

# We will create one environment to evaluate the agent on
eval_env = PEVBatteryChargeCentral(config)

#%%
reward_averages = []
reward_std = []
training_times = []
total_procs = 0
total_procs += n_procs

if n_procs == 1:
    # if there is only one process, there is no need to use multiprocessing
    train_env = DummyVecEnv([lambda: PEVBatteryChargeCentral(config)])
else:
    # Here we use the "fork" method for launching the processes, more information is available in the doc
    # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
    train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)])

rewards = []
times = []

# it is recommended to run several experiments due to variability in results
train_env.reset()
model = ALGO('MlpPolicy', train_env, verbose=1)
start = time.time()
model.learn(total_timesteps=50000)
times.append(time.time() - start)
mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
rewards.append(mean_reward)
# Important: when using subprocess, don't forget to close them
# otherwise, you may have memory issues when running a lot of experiments
train_env.close()
reward_averages.append(np.mean(rewards))
reward_std.append(np.std(rewards))
training_times.append(np.mean(times))
# %%
