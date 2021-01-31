from pyvirtualdisplay import Display
from time import sleep
import imageio
from argparse import Namespace

from pev_battery_charge.envs.config_pev import get_config
from pev_battery_charge.envs import PEVBatteryCharge

#from envs import MPEEnv
# from envs import DronesEnv
#from gym_pybullet_drones_temp.envs.SimpleParticleFormation import SimpleParticleFormation 
from utils.storage import RolloutStorage
from utils.single_storage import SingleRolloutStorage
import torch
import itertools
import numpy as np
from multiprocessing import Process
from datetime import datetime
# replay buffer 

def reset_rollouts(rollouts, envs, args):
    obs, _ = envs.reset()
    if args.share_policy: 
        obs = np.array(obs)
        share_obs = obs.reshape(args.n_rollout_threads, -1)        
        share_obs = np.expand_dims(share_obs,1).repeat(args.num_agents,axis=1)    
        rollouts.share_obs[0] = share_obs.copy() 
        rollouts.obs[0] = obs.copy()               
        rollouts.recurrent_hidden_states = np.zeros(rollouts.recurrent_hidden_states.shape).astype(np.float32)
        rollouts.recurrent_hidden_states_critic = np.zeros(rollouts.recurrent_hidden_states_critic.shape).astype(np.float32)
    else:
        share_obs = []
        for o in obs:
            share_obs.append(list(itertools.chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(args.num_agents):    
            rollouts[agent_id].share_obs[0] = share_obs.copy()
            rollouts[agent_id].obs[0] = np.array(list(obs[:,agent_id])).copy()               
            rollouts[agent_id].recurrent_hidden_states = np.zeros(rollouts[agent_id].recurrent_hidden_states.shape).astype(np.float32)
            rollouts[agent_id].recurrent_hidden_states_critic = np.zeros(rollouts[agent_id].recurrent_hidden_states_critic.shape).astype(np.float32)
    
    return rollouts
    
def gif_assembling(actor_critic, args, gif_name, n_eps = 1):
    
    start = datetime.now()
    
    args = Namespace(**vars(args).copy()) # For security
    args.n_rollout_threads = 1
    num_agents = args.num_agents
    
    if args.env_name == "BatteryCharge":
        envs = PEVBatteryCharge(args)
    # elif args.env_name == "drones":
    #     envs = DronesEnv(args)
    
    images = []
    if args.share_policy:
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)  
    else:
        rollouts = []
        for agent_id in range(num_agents):
            ro = SingleRolloutStorage(agent_id,
                        args.episode_length, 
                        args.n_rollout_threads,
                        envs.observation_space, 
                        envs.action_space,
                        args.hidden_size)
            rollouts.append(ro)
            
    for episode in range(n_eps):
        rollouts = reset_rollouts(rollouts, envs, args)
        for step in range(args.episode_length):
            # Sample actions
            values = []
            actions= []
            action_log_probs = []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []

            with torch.no_grad():                
                for agent_id in range(num_agents):
                    if args.share_policy:
                        actor_critic.eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic.act(agent_id,
                            torch.FloatTensor(rollouts.share_obs[step,:,agent_id]), 
                            torch.FloatTensor(rollouts.obs[step,:,agent_id]), 
                            torch.FloatTensor(rollouts.recurrent_hidden_states[step,:,agent_id]), 
                            torch.FloatTensor(rollouts.recurrent_hidden_states_critic[step,:,agent_id]),
                            torch.FloatTensor(rollouts.masks[step,:,agent_id]))
                    else:
                        actor_critic[agent_id].eval()
                        value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = actor_critic[agent_id].act(agent_id,
                            torch.FloatTensor(rollouts[agent_id].share_obs[step,:]), 
                            torch.FloatTensor(rollouts[agent_id].obs[step,:]), 
                            torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states[step,:]), 
                            torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states_critic[step,:]),
                            torch.FloatTensor(rollouts[agent_id].masks[step,:]))
                        
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    action_log_probs.append(action_log_prob.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
            
            # rearrange action
            actions_env = []
            for i in range(args.n_rollout_threads):
                action_env = []
                for agent_id in range(num_agents):
                    if envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        uc_action = []
                        for j in range(envs.action_space[agent_id].shape):
                            uc_one_hot_action = np.zeros(envs.action_space[agent_id].high[j]+1)
                            uc_one_hot_action[actions[agent_id][i][j]] = 1
                            uc_action.append(uc_one_hot_action)
                        uc_action = np.concatenate(uc_action)
                        action_env.append(uc_action)
                            
                    elif envs.action_space[agent_id].__class__.__name__ == 'Discrete':    
                        one_hot_action = np.zeros(envs.action_space[agent_id].n)
                        one_hot_action[actions[agent_id][i]] = 1
                        action_env.append(one_hot_action)
                        
                    elif envs.action_space[agent_id].__class__.__name__ == 'Box': # Arci Added.
                        action_env.append(actions[agent_id][i])
                    
                    else:
                        raise NotImplementedError
                actions_env.append(action_env)
                
            # Obser reward and next obs
            obs, rewards, dones, infos, _ = envs.step(actions_env[0])
            obs = np.array(obs)
            images.append(envs.render(plots=[1,2,3,4,5], mode='rgb_array'))
            if step % 10 == 0:
                print(str(step)+", ", end='')
            # If done then clean the history of observations.
            # insert data in buffer
            masks = []
            mask = []               
            for agent_id in range(num_agents): 
                if dones[agent_id]:    
                    recurrent_hidden_statess[agent_id][0] = np.zeros(args.hidden_size).astype(np.float32)
                    recurrent_hidden_statess_critic[agent_id][0] = np.zeros(args.hidden_size).astype(np.float32)    
                    mask.append([0.0])
                else:
                    mask.append([1.0])
            masks.append(mask)
                            
            if args.share_policy: 
                share_obs = obs.reshape(args.n_rollout_threads, -1)        
                share_obs = np.expand_dims(share_obs,1).repeat(num_agents,axis=1)    
                
                rollouts.insert(share_obs, 
                            obs, 
                            np.array(recurrent_hidden_statess).transpose(1,0,2), 
                            np.array(recurrent_hidden_statess_critic).transpose(1,0,2), 
                            np.array(actions).transpose(1,0,2),
                            np.array(action_log_probs).transpose(1,0,2), 
                            np.array(values).transpose(1,0,2),
                            rewards, 
                            masks)
            else:
                share_obs = []
                for o in obs:
                    share_obs.append(list(itertools.chain(*o)))
                share_obs = np.array(share_obs)
                for agent_id in range(num_agents):
                    rollouts[agent_id].insert(share_obs, 
                            np.array(list(obs[:,agent_id])), 
                            np.array(recurrent_hidden_statess[agent_id]), 
                            np.array(recurrent_hidden_statess_critic[agent_id]), 
                            np.array(actions[agent_id]),
                            np.array(action_log_probs[agent_id]), 
                            np.array(values[agent_id]),
                            rewards[:,agent_id], 
                            np.array(masks)[:,agent_id])
                    
    imageio.mimsave(gif_name, np.array(images), fps=20)
    print("Gif assembling process finished. Time to completion: ", str(datetime.now() - start))       
    print(">> Gif rendering was successfully saved at: ", gif_name)
    
def generate_gif(actor_critic, args, gif_name, n_eps = 1, parallel=False):
    start = datetime.now()
    
    if parallel:
        print(start, " - Gif assembling process started parallelly.")
        proc = Process(target=gif_assembling, args=(actor_critic, args, gif_name, n_eps))
        proc.start()
        proc.join()
        
    else:
        print(start, " - Gif assembling process started.")
        gif_assembling(actor_critic, args, gif_name, n_eps)
        print("Gif assembling process finished. Time to completion: ", str(datetime.now() - start))
        print(">> Gif rendering was successfully saved at: ", gif_name)
    return str(gif_name)

def load_gif(gif_name):
    from IPython.display import Image
    with open(gif_name,'rb') as f:
        image = Image(data=f.read(), format='png')
    return image