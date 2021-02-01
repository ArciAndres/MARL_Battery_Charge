#%%
import os
import time
import json
import numpy as np
import math
import matplotlib.pyplot as plt

import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from utils_training import reset_rollouts
from utils_training import generate_gif_sb
from pyvirtualdisplay import Display
from pathlib import Path

#%%

from pev_battery_charge.envs.PEVBatteryChargeCentral import PEVBatteryChargeCentral
from pev_battery_charge.envs.config_pev import get_config

# Print results
from tabulate import tabulate
pdtable = lambda df:tabulate(df,tablefmt='psql')
#%%
def make_env(args, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PEVBatteryChargeCentral(args)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init
#%%
from stable_baselines3.common.cmd_util import make_vec_env

def main():
    args = get_config()

    #%%
    algos = {"PPO": PPO, "A2C": A2C, "TD3": TD3}
    ALGO = algos[args.algorithm_name]


    # =============== Model restore ===============   
    restore = args.restore_model
    
    # if restore:
    #     assert args.restore_model is not None, "A path from a previously saved model must be indicated."
    
    #     restore_folder = Path(args.restore_model_path) / 'models'
    #     with open( restore_folder / 'train_checkpoint.json' , 'r') as f:
    #         train_checkpoint = json.load(f)
    #     with open( restore_folder / 'config.json' , 'r') as f:
    #         config_import = json.load(f)
    
    #     curr_run = train_checkpoint['curr_run']
    #     ET_restore = train_checkpoint['ET']
    #     episode_restore = train_checkpoint['episode']
    #     args = Namespace(**config_import)
    
    # else:
    #     ET_restore = 0
    #     episode_restore = 0
    # =============== Model restore ===============

    # We will create one environment to evaluate the agent on
    env_eval = PEVBatteryChargeCentral(args)

    if args.n_rollout_threads == 1:
        # if there is only one process, there is no need to use multiprocessing
        env = DummyVecEnv([lambda: PEVBatteryChargeCentral(args)])
    else:
        # Here we use the "fork" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        env = SubprocVecEnv([make_env(args, i) for i in range(args.n_rollout_threads)])

    rewards = []
    times = []

    #============ Folder creation ==================
        # path
    model_dir = Path('./results') / args.scenario_name / args.algorithm_name
    
    if not restore:
        if not model_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    save_dir = run_dir / 'models'
    gif_dir = run_dir / 'media'

    if not restore:
        os.makedirs(str(log_dir))
        os.makedirs(str(save_dir))
        os.makedirs(str(gif_dir))

    #============== Display and time ====================
    if args.save_gifs and args.colab:
        display = Display(visible=0, size=(1400, 900))
        display.start()
        print(">>> VIRTUAL DISPLAY STARTED")
    
    start = time.time()
    
    args.episode_length = math.floor((args.total_time/args.sampling_time))
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    iterations = args.iterations
    
    print("============================  TRAINING BEGINS ============================\n")
    print("Run directory:" , run_dir, '\n')
    train_checkpoint = { 'curr_run': curr_run }
    
    # Print and save configuration for this run
    args_dir = vars(args)
    print(pdtable([args_dir.keys(), args_dir.values()]))
    
    with open(save_dir / 'config.json', "w") as f:
        json.dump(vars(args), f)

    #============= Training ========================
    eps = 1
    model = ALGO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    print("============ TRAINING STARTS =============")

    for iteration in range(0, iterations):
        start_it = time.time()
        print(">> General iteration: ", iteration)
        model.learn(total_timesteps=args.num_env_steps,learning_rate=args.lr)
        print("Duration of iteration: ", time.time() - start_it)
        print("Total duration: ", time.time()- start)
        gif_name = gif_dir / ("step"+ str(iteration)+".gif")
        generate_gif_sb(model, env_eval, args, gif_name=gif_name, n_eps=1 )

        mean_reward, std_reward  = evaluate_policy(model, env_eval, n_eval_episodes=args.eval_episodes)
        
        print("Mean reward: ", mean_reward, " +/- ", std_reward)
    # %%
if __name__ == "__main__":
    main()