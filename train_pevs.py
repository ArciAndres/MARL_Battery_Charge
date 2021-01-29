import os
import time
import numpy as np
from pathlib import Path
import json

import torch
from tensorboardX import SummaryWriter

#from envs import MPEEnv

from algorithm.ppo import PPO
from algorithm.model import Policy

from pev_battery_charge.envs.config_pev import get_config
from pev_battery_charge.envs import PEVBatteryCharge

#from config import get_config
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.util import update_linear_schedule
from utils.storage import RolloutStorage
from utils.single_storage import SingleRolloutStorage
import numpy as np
import itertools
import os
from pdb import set_trace
import math

# Save gifs imports
from pyvirtualdisplay import Display
from time import sleep
from argparse import Namespace
from utils_training import generate_gif

# Print results
from tabulate import tabulate
pdtable = lambda df:tabulate(df,tablefmt='psql')

from utils_training import reset_rollouts

#%%

def make_parallel_env(args):
    def get_env_fn(rank):
        def init_env():
            if args.env_name == "BatteryCharge":
                env = PEVBatteryCharge(args)
            else:
                print("Can not support the " + args.env_name + "environment." )
                raise NotImplementedError
            env.set_seed(args.seed + rank * 1000)
            # np.random.seed(args.seed + rank * 1000)
            return env
        return init_env
    if args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout_threads)])
#%%
def main():
    args = get_config()
    # set_trace()
    # #=========== Restore model =========
    # args.restore_model = True
    # args.restore_model_path = 'results/drones/simple_formation_1/mappo_gru/run10'
    
    restore = args.restore_model
    
    if restore:
        assert args.restore_model is not None, "A path from a previously saved model must be indicated."
    
        restore_folder = Path(args.restore_model_path) / 'models'
        with open( restore_folder / 'train_checkpoint.json' , 'r') as f:
            train_checkpoint = json.load(f)
        with open( restore_folder / 'config.json' , 'r') as f:
            config_import = json.load(f)
    
        curr_run = train_checkpoint['curr_run']
        ET_restore = train_checkpoint['ET']
        episode_restore = train_checkpoint['episode']
        args = Namespace(**config_import)
    
    else:
        ET_restore = 0
        episode_restore = 0
    
    #-----------------------------------
    
    assert (args.share_policy == True and args.scenario_name == 'simple_speaker_listener') == False, ("The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")
    
    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # cuda
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)
    
    
    # path
    model_dir = Path('./results') / args.env_name / args.scenario_name / args.algorithm_name
    
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
    
    args.episode_length = math.floor((args.total_time/args.sampling_time))
    
    if not restore:
        os.makedirs(str(log_dir))
        os.makedirs(str(save_dir))
        os.makedirs(str(gif_dir))
    logger = SummaryWriter(str(log_dir)) 
    #%%
    # env
    #set_trace()
    
    envs = make_parallel_env(args)
    num_agents = args.num_agents
    
    #Policy network
    
    if args.share_policy:
        if restore:
            actor_critic = torch.load(save_dir / 'agent_model.pt')['model']
            print(">> Loaded saved model: ", save_dir / 'agent_model.pt')
        else:
            actor_critic = Policy(envs.observation_space[0], 
                        envs.action_space[0],
                        num_agents = num_agents,
                        gain = args.gain,
                        base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                    'recurrent': args.recurrent_policy,
                                    'hidden_size': args.hidden_size,
                                    'recurrent_N': args.recurrent_N,
                                    'attn': args.attn,       
                                    'attn_only_critic': args.attn_only_critic,                           
                                    'attn_size': args.attn_size,
                                    'attn_N': args.attn_N,
                                    'attn_heads': args.attn_heads,
                                    'dropout': args.dropout,
                                    'use_average_pool': args.use_average_pool,
                                    'use_common_layer':args.use_common_layer,
                                    'use_feature_normlization':args.use_feature_normlization,
                                    'use_feature_popart':args.use_feature_popart,
                                    'use_orthogonal':args.use_orthogonal,
                                    'layer_N':args.layer_N,
                                    'use_ReLU':args.use_ReLU,
                                    'use_same_dim':args.use_same_dim
                                    },
                        device = device)
        actor_critic.to(device)
        # algorithm
        agents = PPO(actor_critic,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   use_value_high_masks=args.use_value_high_masks,
                   device=device)
                   
        #replay buffer
        rollouts = RolloutStorage(num_agents,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space[0], 
                    envs.action_space[0],
                    args.hidden_size)        
    else:
        actor_critic = []
        agents = []
        rollouts = []
        for agent_id in range(num_agents):
            if restore:
                ac = torch.load(save_dir / ("agent%i_model" % agent_id + ".pt"))['model']
            else:
                ac = Policy(envs.observation_space, 
                          envs.action_space[agent_id],
                          num_agents = agent_id, # here is special
                          gain = args.gain,
                          base_kwargs={'naive_recurrent': args.naive_recurrent_policy,
                                    'recurrent': args.recurrent_policy,
                                    'hidden_size': args.hidden_size,
                                    'recurrent_N': args.recurrent_N,
                                    'attn': args.attn,  
                                    'attn_only_critic': args.attn_only_critic,                                
                                    'attn_size': args.attn_size,
                                    'attn_N': args.attn_N,
                                    'attn_heads': args.attn_heads,
                                    'dropout': args.dropout,
                                    'use_average_pool': args.use_average_pool,
                                    'use_common_layer':args.use_common_layer,
                                    'use_feature_normlization':args.use_feature_normlization,
                                    'use_feature_popart':args.use_feature_popart,
                                    'use_orthogonal':args.use_orthogonal,
                                    'layer_N':args.layer_N,
                                    'use_ReLU':args.use_ReLU,
                                    'use_same_dim':args.use_same_dim
                                    },
                          device = device)
            ac.to(device)
            # algorithm
            agent = PPO(ac,
                   args.clip_param,
                   args.ppo_epoch,
                   args.num_mini_batch,
                   args.data_chunk_length,
                   args.value_loss_coef,
                   args.entropy_coef,
                   logger,
                   lr=args.lr,
                   eps=args.eps,
                   weight_decay=args.weight_decay,
                   max_grad_norm=args.max_grad_norm,
                   use_max_grad_norm=args.use_max_grad_norm,
                   use_clipped_value_loss= args.use_clipped_value_loss,
                   use_common_layer=args.use_common_layer,
                   use_huber_loss=args.use_huber_loss,
                   huber_delta=args.huber_delta,
                   use_popart=args.use_popart,
                   use_value_high_masks=args.use_value_high_masks,
                   device=device)
                               
            actor_critic.append(ac)
            agents.append(agent) 
              
            #replay buffer
            ro = SingleRolloutStorage(agent_id,
                    args.episode_length, 
                    args.n_rollout_threads,
                    envs.observation_space, 
                    envs.action_space,
                    args.hidden_size)
            rollouts.append(ro)
    
    rollouts = reset_rollouts(rollouts, envs, args)
    
    #%%
    if args.save_gifs and args.colab:
        display = Display(visible=0, size=(1400, 900))
        display.start()
        print(">>> VIRTUAL DISPLAY STARTED")
    
    start = time.time()
    
    episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
    
    
    print("============================  TRAINING BEGINS ============================\n")
    print("Run directory:" , run_dir, '\n')
    train_checkpoint = { 'curr_run': curr_run }
    
    # Print and save configuration for this run
    args_dir = vars(args)
    print(pdtable([args_dir.keys(), args_dir.values()]))
    
    with open(save_dir / 'config.json', "w") as f:
        json.dump(vars(args), f)
    
    for episode in range(episode_restore, episodes):
        if args.use_linear_lr_decay:# decrease learning rate linearly
            if args.share_policy:   
                update_linear_schedule(agents.optimizer, episode, episodes, args.lr)  
            else:     
                for agent_id in range(num_agents):
                    update_linear_schedule(agents[agent_id].optimizer, episode, episodes, args.lr)           
    
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
            obs, rewards, dones, infos, _ = envs.step(actions_env)
            
            obs = np.array(obs)
            # If done then clean the history of observations.
            # insert data in buffer
            masks = []
            for i, done in enumerate(dones): 
                mask = []               
                for agent_id in range(num_agents): 
                    if done[agent_id]:    
                        recurrent_hidden_statess[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)
                        recurrent_hidden_statess_critic[agent_id][i] = np.zeros(args.hidden_size).astype(np.float32)    
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
    #%%
    
        with torch.no_grad(): 
            for agent_id in range(num_agents):         
                if args.share_policy: 
                    actor_critic.eval()                
                    next_value,_,_ = actor_critic.get_value(agent_id,
                                                    torch.FloatTensor(rollouts.share_obs[-1,:,agent_id]), 
                                                    torch.FloatTensor(rollouts.obs[-1,:,agent_id]), 
                                                    torch.FloatTensor(rollouts.recurrent_hidden_states[-1,:,agent_id]),
                                                    torch.FloatTensor(rollouts.recurrent_hidden_states_critic[-1,:,agent_id]),
                                                    torch.FloatTensor(rollouts.masks[-1,:,agent_id]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts.compute_returns(agent_id,
                                    next_value, 
                                    args.use_gae, 
                                    args.gamma,
                                    args.gae_lambda, 
                                    args.use_proper_time_limits,
                                    args.use_popart,
                                    agents.value_normalizer)
                else:
                    actor_critic[agent_id].eval()
                    next_value,_,_ = actor_critic[agent_id].get_value(agent_id,
                                                              torch.FloatTensor(rollouts[agent_id].share_obs[-1,:]), 
                                                              torch.FloatTensor(rollouts[agent_id].obs[-1,:]), 
                                                              torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states[-1,:]),
                                                              torch.FloatTensor(rollouts[agent_id].recurrent_hidden_states_critic[-1,:]),
                                                              torch.FloatTensor(rollouts[agent_id].masks[-1,:]))
                    next_value = next_value.detach().cpu().numpy()
                    rollouts[agent_id].compute_returns(next_value, 
                                            args.use_gae, 
                                            args.gamma,
                                            args.gae_lambda, 
                                            args.use_proper_time_limits,
                                            args.use_popart,
                                            agents[agent_id].value_normalizer)
        
        # update the network
        if args.share_policy:
            actor_critic.train()
            value_loss, action_loss, dist_entropy = agents.update_share(num_agents, rollouts)
                            
            for agent_id in range(num_agents):
                rew = []
                for i in range(rollouts.rewards.shape[1]):
                    rew.append(np.sum(rollouts.rewards[:,i,agent_id]))
                logger.add_scalars('agent%i/average_episode_reward' % agent_id,
                    {'average_episode_reward': np.mean(rew)},
                    (episode + 1) * args.episode_length * args.n_rollout_threads)
            # clean the buffer and reset
            rollouts.after_update()
        else:
            value_losses = []
            action_losses = []
            dist_entropies = [] 
            
            for agent_id in range(num_agents):
                actor_critic[agent_id].train()
                value_loss, action_loss, dist_entropy = agents[agent_id].update_single(agent_id, rollouts[agent_id])
                value_losses.append(value_loss)
                action_losses.append(action_loss)
                dist_entropies.append(dist_entropy)
                    
                rew = []
                for i in range(rollouts[agent_id].rewards.shape[1]):
                    rew.append(np.sum(rollouts[agent_id].rewards[:,i]))
                logger.add_scalars('agent%i/average_episode_reward'%agent_id,
                    {'average_episode_reward': np.mean(rew)},
                    (episode + 1) * args.episode_length * args.n_rollout_threads)
                
                rollouts[agent_id].after_update()
                                                                      
        total_num_steps = (episode + 1) * args.episode_length * args.n_rollout_threads
    
        if (episode % args.save_interval == 0 or episode == episodes - 1):# save for every interval-th episode or for the last epoch
            if args.share_policy:
                torch.save({
                            'model': actor_critic
                            }, 
                            str(save_dir) + "/agent_model.pt")
            else:
                for agent_id in range(num_agents):                                                  
                    torch.save({
                                'model': actor_critic[agent_id]
                                }, 
                                str(save_dir) + "/agent%i_model" % agent_id + ".pt")
    
            # Save checkpoint in case of restore
            train_checkpoint['episode'] = episode
            train_checkpoint['ET'] = time.time() - start
    
            with open(run_dir / 'models' / 'train_checkpoint.json', 'w') as f:
                json.dump(train_checkpoint, f)
        if args.save_gifs:
            if (episode % args.save_gifs_interval == 0 or episode == episodes - 1):
                print("Saving gif...")
                gif_name = gif_dir / ("step"+ str(episode)+".gif")
                
                generate_gif(actor_critic, args, gif_name, n_eps=3, parallel=args.parallel_gif)
    
    
        # log information
        if episode % args.log_interval == 0:
            end = time.time()
            FPS = total_num_steps / (end - start + ET_restore)
            table_ = [['Scenario', 'Algorithm', 'Episodes', 'Total timesteps', 'FPS', 'ET (m)', 'ETA (m)'], 
                     [args.scenario_name, 
                      args.algorithm_name,
                      "%d/%d" % (episode,episodes),
                      "%d/%d" % (total_num_steps , args.num_env_steps),
                      int(FPS),
                      round((end - start + ET_restore)/60, 2), 
                      round((args.num_env_steps - total_num_steps)/FPS/60, 2)
                      ]
                     ]
            if args.share_policy:
              table_[0].append('Loss of agent')
              table_[1].append(round(value_loss, 3))
            else:
              for agent_id in range(num_agents):
                table_[0].append('Loss of agent%i')
                table_[1].append(round(value_losses[agent_id], 3))
            
            if args.env_name == "MPE" or args.env_name == 'BatteryCharge':
                show_rewards = [i['rewards_info_sum'] for i in infos]
                show_rewards = np.mean(show_rewards, axis=0)
                
                for agent_id in range(num_agents):
                    table_[0].append('Reward (%s)' % agent_id)
                    table_[1].append(round(show_rewards[agent_id], 3))
                    logger.add_scalars('agent%i/individual_reward' % agent_id, {'individual_reward': show_rewards[agent_id]}, total_num_steps)
                
                logger.add_scalars('joint/reward', {'joint_reward': np.mean(show_rewards)}, total_num_steps)
                
            if episode % args.log_console == 0:
                print(pdtable(table_))
                
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    envs.close()
    print("===================== TRAINING FINISHED ======================")

if __name__ == "__main__":
    main()
    
