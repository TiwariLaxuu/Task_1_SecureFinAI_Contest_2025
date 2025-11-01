import os
import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from erl_config import Config
from erl_replay_buffer import ReplayBuffer
from erl_net import QNetTwin, QNetTwinDuel, ActorDiscretePPO, CriticPPO


def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


class AgentDoubleDQN:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        '''optimizer'''
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

        self.act_target = self.cri_target = deepcopy(self.act)
        self.act.explore_rate = getattr(args, "explore_rate", 1 / 32)

    def get_obj_critic(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)

            next_qs = torch.min(*self.cri_target.get_q1_q2(next_ss)).max(dim=1, keepdim=True)[0].squeeze(1)
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = [qs.gather(1, actions.long()).squeeze(1) for qs in self.act.get_q1_q2(states)]
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, q1

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, torch.load(file_path, map_location=self.device))

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

        # get_action = self.act.get_action # TODO check
        get_action = self.act_target.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1)) if if_random \
                else get_action(state).detach()  # different
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_qs = self.cri_target(next_ss).max(dim=1, keepdim=True)[0].squeeze(1)  # next q_values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states).gather(1, actions.long()).squeeze(1)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, q_values

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = self.act_target(last_state).argmax(dim=1).detach()  # actor is Q Network in DQN style
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

class AgentTwinD3QN(AgentDoubleDQN):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)


import os
import torch
import torch.nn as nn
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from torch.distributions import Normal, Categorical
from torch.nn.utils import clip_grad_norm_

from erl_config import Config
from erl_replay_buffer import ReplayBuffer


class AgentPPO:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.num_envs = args.num_envs
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = getattr(args, "if_off_policy", False)
        self.clip_grad_norm = args.clip_grad_norm
        self.soft_update_tau = args.soft_update_tau
        self.state_value_tau = args.state_value_tau
        
        # PPO specific parameters
        self.lambda_gae = getattr(args, "lambda_gae", 0.95)
        self.ratio_clip = getattr(args, "ratio_clip", 0.2)
        self.entropy_coef = getattr(args, "entropy_coef", 0.02)
        self.kl_coef = getattr(args, "kl_coef", 0.5)
        self.kl_target = getattr(args, "kl_target", 0.01)
        
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.last_state = None  # last state of the trajectory for training
        
        '''network'''
        self.act = self.act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_class(net_dims, state_dim).to(self.device) \
            if hasattr(self, "cri_class") else self.act
            
        '''optimizer'''
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if hasattr(self, "cri_class") else self.act_optimizer
            
        self.criterion = nn.MSELoss()
        
        """save and load"""
        self.save_attr_names = {'act', 'cri', 'act_optimizer', 'cri_optimizer'}

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction.
        
        Returns:
            states: shape (horizon_len, num_envs, state_dim)
            actions: shape (horizon_len, num_envs, action_dim)
            rewards: shape (horizon_len, num_envs)
            undones: shape (horizon_len, num_envs)
            logprobs: shape (horizon_len, num_envs)
            values: shape (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        logprobs = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        values = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        
        state = self.last_state
        get_action = self.act.get_action
        for t in range(horizon_len):
#             action = torch.randint(self.action_dim, size=(self.num_envs, 1)) if if_random \
#                 else get_action(state).detach()
            action, logprob = self.act.get_action_logprob(state)
            value = self.cri(state)
            
            next_state, reward, done, _ = env.step(action)
            
            states[t] = state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            logprobs[t] = logprob
            values[t] = value
            
            state = next_state
            
        self.last_state = state
        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        
        return states, actions, rewards, undones, logprobs, values

    def compute_advantages(self, rewards: Tensor, values: Tensor, undones: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: shape (horizon_len, num_envs)
            returns: shape (horizon_len, num_envs)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        masks = undones * self.gamma
        horizon_len = rewards.shape[0]
        
        last_state = self.last_state
        next_value = self.cri(last_state).detach()
        
        gae = 0
        for t in reversed(range(horizon_len)):
            delta = rewards[t] + masks[t] * next_value - values[t]
            gae = delta + masks[t] * self.lambda_gae * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        Update the actor and critic networks using PPO.
        
        Returns:
            obj_critic: critic loss
            obj_actor: actor loss
            kl: KL divergence
            entropy: entropy of the policy
        """
        states, actions, rewards, undones, old_logprobs, old_values = buffer.sample()
        advantages, returns = self.compute_advantages(rewards, old_values, undones)
        
        obj_critics = 0.0
        obj_actors = 0.0
        kls = 0.0
        entropies = 0.0
        update_times = 2
#         try:
#             update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
#         except:
#             update_times = 2
        assert update_times >= 1
        
        for _ in range(update_times):
            indices = torch.randint(0, 10, size=(self.batch_size,))
            
            # Get batch data
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_logprobs = old_logprobs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]
            
            # Get new action probabilities and values
            new_logprobs, entropy = self.act.get_logprob_entropy(batch_states, batch_actions)
            new_values = self.cri(batch_states)
            
            # Calculate ratios
            log_ratios = new_logprobs - batch_old_logprobs
            ratios = log_ratios.exp()
            
            # Policy loss
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip) * batch_advantages
            obj_actor = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            
            # Value loss
            obj_critic = self.criterion(new_values, batch_returns)
            
            # KL divergence
            with torch.no_grad():
                kl = ((ratios.log() - log_ratios) * ratios).mean()
                kls += kl.item()
                
            # Optimize
            self.optimizer_update(self.act_optimizer, obj_actor)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            
            obj_actors += obj_actor.item()
            obj_critics += obj_critic.item()
            entropies += entropy.mean().item()
            
        return (obj_critics / update_times, 
                obj_actors / update_times)
#                 kls / update_times, 
#                 entropies / update_times)


    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent"""
        assert self.save_attr_names.issuperset({'act', 'cri', 'act_optimizer', 'cri_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, torch.load(file_path, map_location=self.device))

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network"""
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective"""
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()


class AgentPPOContinuous(AgentPPO):
    """PPO for continuous action spaces"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", ActorPPO)  # Gaussian policy
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)


class AgentPPODiscrete(AgentPPO):
    """PPO for discrete action spaces"""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", ActorDiscretePPO)  # Categorical policy
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)