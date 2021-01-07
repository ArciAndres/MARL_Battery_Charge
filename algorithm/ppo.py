import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

def huber_loss(e, d):
    a = (abs(e)<=d).float()
    b = (e>d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)
    
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:continue
        sum_grad += x.grad.norm() ** 2    
    return math.sqrt(sum_grad)

class PopArt(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.train = True
        self.device = device

        self.running_mean = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).to(self.device)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).to(self.device)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False).to(self.device)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector):
        # Make sure input is float32
        input_vector = input_vector.to(torch.float).to(self.device)

        if self.train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()            
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        input_vector = input_vector.to(torch.float).to(self.device)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out

class PPO():
    def __init__(self,                 
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 data_chunk_length,
                 value_loss_coef,
                 entropy_coef,
                 logger = None,
                 lr=None,
                 eps=None,
                 weight_decay=None,
                 max_grad_norm=None,
                 use_max_grad_norm=True,
                 use_clipped_value_loss=True,
                 use_common_layer=False,
                 use_huber_loss = False,
                 huber_delta=2,
                 use_popart = True,
                 use_value_high_masks = False,
                 device = torch.device("cpu")):

        self.step=0
        self.device = device
        self.logger = logger
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.data_chunk_length = data_chunk_length

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_max_grad_norm = use_max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_common_layer = use_common_layer
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.use_popart = use_popart
        self.use_value_high_masks = use_value_high_masks
        if self.use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def update_single(self, agent_id, rollouts, turn_on=True):
        if self.use_popart:
            advantages = rollouts.returns[:-1,:] - self.value_normalizer.denormalize(torch.tensor(rollouts.value_preds[:-1,:])).cpu().numpy()
        else:
            advantages = rollouts.returns[:-1,:] - rollouts.value_preds[:-1,:]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                     advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator(
                     advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                        
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                adv_targ = adv_targ.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                high_masks_batch = high_masks_batch.to(self.device)
                
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(agent_id, share_obs_batch, 
                obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, masks_batch, high_masks_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                
                KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch, torch.exp(action_log_probs))
                
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ                
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch).sum() / high_masks_batch.sum()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = self.value_normalizer(return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum()
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = (return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = (return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum()
                    else:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - self.value_normalizer(return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch)).pow(2)
                            value_loss = 0.5 * ( (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum() )
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - (return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                            value_loss = 0.5 * ( (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum() )
                    
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch) - values
                        else:
                            error = return_batch - values
                        value_loss = (huber_loss(error, self.huber_delta) * high_masks_batch).sum() / high_masks_batch.sum()
                    else:
                        if self.use_popart:
                            value_loss = 0.5 * (((self.value_normalizer(return_batch) - values).pow(2) * high_masks_batch).sum() / high_masks_batch.sum())
                        else:
                            value_loss = 0.5 * (((return_batch - values).pow(2) * high_masks_batch).sum() / high_masks_batch.sum())
                
                self.optimizer.zero_grad()
                
                if self.use_common_layer:
                    (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                else:              
                    (value_loss * self.value_loss_coef).backward()
                    if turn_on == True:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                
                grad_norm = get_gard_norm(self.actor_critic.parameters())
                       
                if self.use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                if self.logger is not None:
                    self.logger.add_scalars('agent%i/value_loss' % agent_id,
                        {'value_loss': value_loss},
                        self.step)
                    self.logger.add_scalars('agent%i/action_loss' % agent_id,
                        {'action_loss': action_loss},
                        self.step)
                    self.logger.add_scalars('agent%i/dist_entropy' % agent_id,
                        {'dist_entropy': dist_entropy},
                        self.step)
                    self.logger.add_scalars('agent%i/KL_divloss' % agent_id,
                        {'KL_divloss': KL_divloss},
                        self.step)
                    self.logger.add_scalars('agent%i/grad_norm' % agent_id,
                        {'grad_norm': grad_norm},
                        self.step)
                    self.step += 1

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()                
                
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
        
    def update(self, agent_id, rollouts, turn_on=True):
        if self.use_popart:
            advantages = rollouts.returns[:-1,:,agent_id] - self.value_normalizer.denormalize(torch.tensor(rollouts.value_preds[:-1,:,agent_id])).cpu().numpy()
        else:
            advantages = rollouts.returns[:-1,:,agent_id] - rollouts.value_preds[:-1,:,agent_id]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    agent_id, advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator(
                    agent_id, advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    agent_id, advantages, self.num_mini_batch)

            for sample in data_generator:
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                        
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                adv_targ = adv_targ.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                high_masks_batch = high_masks_batch.to(self.device)
                
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(agent_id, share_obs_batch, 
                obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, masks_batch, high_masks_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                
                KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch, torch.exp(action_log_probs))
                
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ                
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch).sum() / high_masks_batch.sum()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = self.value_normalizer(return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum()
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = (return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = (return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            value_loss = (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum()
                    else:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - self.value_normalizer(return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch)).pow(2)
                            value_loss = 0.5 * ( (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum() )
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - (return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                            value_loss = 0.5 * ( (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum() )
                    
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch) - values
                        else:
                            error = return_batch - values
                        value_loss = (huber_loss(error, self.huber_delta) * high_masks_batch).sum() / high_masks_batch.sum()
                    else:
                        if self.use_popart:
                            value_loss = 0.5 * (((self.value_normalizer(return_batch) - values).pow(2) * high_masks_batch).sum() / high_masks_batch.sum())
                        else:
                            value_loss = 0.5 * (((return_batch - values).pow(2) * high_masks_batch).sum() / high_masks_batch.sum())
                
                self.optimizer.zero_grad()
                
                if self.use_common_layer:
                    (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                else:              
                    (value_loss * self.value_loss_coef).backward()
                    if turn_on == True:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
                
                grad_norm = get_gard_norm(self.actor_critic.parameters())
                       
                if self.use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                if self.logger is not None:
                    self.logger.add_scalars('agent%i/value_loss' % agent_id,
                        {'value_loss': value_loss},
                        self.step)
                    self.logger.add_scalars('agent%i/action_loss' % agent_id,
                        {'action_loss': action_loss},
                        self.step)
                    self.logger.add_scalars('agent%i/dist_entropy' % agent_id,
                        {'dist_entropy': dist_entropy},
                        self.step)
                    self.logger.add_scalars('agent%i/KL_divloss' % agent_id,
                        {'KL_divloss': KL_divloss},
                        self.step)
                    self.logger.add_scalars('agent%i/grad_norm' % agent_id,
                        {'grad_norm': grad_norm},
                        self.step)
                    self.step += 1

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()                
                
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_share(self, num_agents, rollouts, turn_on=True):
        advantages = []
        for agent_id in range(num_agents):
            if self.use_popart:
                advantage = rollouts.returns[:-1,:,agent_id] - self.value_normalizer.denormalize(torch.tensor(rollouts.value_preds[:-1,:,agent_id])).cpu().numpy()
            else:
                advantage = rollouts.returns[:-1,:,agent_id] - rollouts.value_preds[:-1,:,agent_id]           
            advantages.append(advantage)
        #agent ,step, parallel,1
        advantages = np.array(advantages).transpose(1,2,0,3)
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)      

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator_share(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.actor_critic.is_naive_recurrent:
                data_generator = rollouts.naive_recurrent_generator_share(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator_share(
                    advantages, self.num_mini_batch)

            for sample in data_generator: 
                share_obs_batch, obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, high_masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample                
                  
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                
                adv_targ = adv_targ.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                high_masks_batch = high_masks_batch.to(self.device)
  
                # Reshape to do in a single forward pass for all steps
                
                values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(agent_id, share_obs_batch, 
                obs_batch, recurrent_hidden_states_batch, recurrent_hidden_states_critic_batch, actions_batch, masks_batch, None)
                
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                KL_divloss = nn.KLDivLoss(reduction='batchmean')(old_action_log_probs_batch, torch.exp(action_log_probs))
                

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = (-torch.min(surr1, surr2)* high_masks_batch).sum() / high_masks_batch.sum()

                if self.use_clipped_value_loss:
                    if self.use_huber_loss:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = self.value_normalizer(return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            if self.use_value_high_masks:
                                value_loss = (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum()
                            else:
                                value_loss = (torch.max(value_losses, value_losses_clipped)).mean()
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            error_clipped = (return_batch) - value_pred_clipped
                            value_losses_clipped = huber_loss(error_clipped, self.huber_delta)
                            error = (return_batch) - values
                            value_losses = huber_loss(error,self.huber_delta)
                            if self.use_value_high_masks:
                                value_loss = (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum()
                            else:
                                value_loss = (torch.max(value_losses, value_losses_clipped)).mean()
                    else:
                        if self.use_popart:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - self.value_normalizer(return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - self.value_normalizer(return_batch)).pow(2)
                            if self.use_value_high_masks:
                                value_loss = 0.5 * ( (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum() )
                            else:
                                value_loss = 0.5 * (torch.max(value_losses, value_losses_clipped)).mean() 
                        else:
                            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                            value_losses = (values - (return_batch)).pow(2)
                            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                            if self.use_value_high_masks:
                                value_loss = 0.5 * ( (torch.max(value_losses, value_losses_clipped) * high_masks_batch).sum() / high_masks_batch.sum() )
                            else:
                                value_loss = 0.5 * (torch.max(value_losses, value_losses_clipped)).mean()
                    
                else:
                    if self.use_huber_loss:
                        if self.use_popart:
                            error = self.value_normalizer(return_batch) - values
                        else:
                            error = return_batch - values
                        if self.use_value_high_masks:
                            value_loss = (huber_loss(error, self.huber_delta) * high_masks_batch).sum() / high_masks_batch.sum()
                        else:
                            value_loss = (huber_loss(error, self.huber_delta)).mean()
                    else:
                        if self.use_popart:
                            if self.use_value_high_masks:
                                value_loss = 0.5 * (((self.value_normalizer(return_batch) - values).pow(2) * high_masks_batch).sum() / high_masks_batch.sum())
                            else:
                                value_loss = 0.5 * (self.value_normalizer(return_batch) - values).pow(2).mean()
                        else:
                            if self.use_value_high_masks:
                                value_loss = 0.5 * (((return_batch - values).pow(2) * high_masks_batch).sum() / high_masks_batch.sum())
                            else:
                                value_loss = 0.5 * (return_batch - values).pow(2).mean() 
                self.optimizer.zero_grad()                 
 
                if self.use_common_layer:
                    (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                else:              
                    (value_loss * self.value_loss_coef).backward()
                    if turn_on == True:
                        (action_loss - dist_entropy * self.entropy_coef).backward()
               
                grad_norm = get_gard_norm(self.actor_critic.parameters())
                       
                if self.use_max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                 
                self.optimizer.step()
                
                if self.logger is not None:
                    self.logger.add_scalars('ratio',
                        {'ratio': ratio.mean()},
                        self.step)
                    self.logger.add_scalars('adv',
                        {'adv': adv_targ.mean()},
                        self.step)
                    self.logger.add_scalars('value_loss',
                        {'value_loss': value_loss},
                        self.step)
                    self.logger.add_scalars('action_loss',
                        {'action_loss': action_loss},
                        self.step)
                    self.logger.add_scalars('dist_entropy',
                        {'dist_entropy': dist_entropy},
                        self.step)
                    self.logger.add_scalars('KL_divloss',
                        {'KL_divloss': KL_divloss},
                        self.step)
                    self.logger.add_scalars('grad_norm',
                        {'grad_norm': grad_norm},
                        self.step)
                    self.step += 1
                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()  
       
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
