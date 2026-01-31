"""
https://github.com/HaozheTian/Torch-PPO-Lagrangian
"""

import scipy
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.nn.functional import softplus
from torch.optim import Adam

from typing import Dict
from typing import Union


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Compute discounted cumulative sums
    Input:
        x = [x0, x1, x2]
    Output:
        [x0 + d * x1 + d^2 * x2, x1 + d * x2, x2]"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def gae_lambda(rews: np.ndarray, vals: np.ndarray, gamma: float, lam: float = 0.97) -> np.ndarray:
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    return discount_cumsum(deltas, gamma * lam)


def obs2ten(x: np.ndarray, target_device: torch.device) -> torch.Tensor:
    return torch.Tensor(x).unsqueeze(0).to(target_device)


def ten2arr(x: torch.Tensor) -> np.ndarray:
    return x.squeeze(0).detach().cpu().numpy()


class Logger:
    def __init__(self):
        self.eps_rets = []
        self.eps_costs = []
        self.eps_lens = []

    def add(self, eps_info: Dict):
        self.eps_rets.append(eps_info['eps_ret'])
        self.eps_costs.append(eps_info['eps_cost'])
        self.eps_lens.append(eps_info['eps_len'])

    def mean(self, var: str) -> float:
        if hasattr(self, var):
            values = getattr(self, var)
            return sum(values) / len(values) if values else 0
        else:
            raise AttributeError(f"'Logger' object has no attribute '{var}'")


class GaussianPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs, act=None):
        mean = self.mlp(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        if act == None:
            act = dist.sample()
            a = dist.log_prob(act)
            log_prob_act = dist.log_prob(act).sum(axis=-1)
            return act, log_prob_act, mean
        else:
            return dist.log_prob(act).sum(axis=-1)


class Value(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        return self.mlp(obs)


class PPOBufferItem:
    def __init__(self, device: torch.device, observations: np.ndarray, actions: np.ndarray,
                 log_probs: np.ndarray, reward_to_gos: np.ndarray, advantages: np.ndarray):
        self.observations = torch.tensor(np.array(observations), dtype=torch.float32, device=device)
        self.actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        self.log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
        self.reward_to_gos = torch.tensor(np.array(reward_to_gos), dtype=torch.float32, device=device)
        self.advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)


class PPOLagBufferItem(PPOBufferItem):
    def __init__(self, device: torch.device, observations: np.ndarray, actions: np.ndarray,
                 log_probs: np.ndarray, reward_to_gos: np.ndarray, advantages: np.ndarray,
                 cost_to_gos: np.ndarray, advantages_cost: np.ndarray):
        super().__init__(device, observations, actions, log_probs, reward_to_gos, advantages)
        self.cost_to_gos = torch.tensor(np.array(cost_to_gos), dtype=torch.float32, device=device)
        self.advantages_cost = torch.tensor(np.array(advantages_cost), dtype=torch.float32, device=device)


class PPOBuffer:
    def __init__(self, device: torch.device, gamma: float = 0.99) -> None:
        self.device = device
        self.gamma = gamma
        self._set_buffers()

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float):
        self._add_transition(obs, act, rew, val, logp)
        self.ptr += 1

    def path_done(self, last_val: float):
        """
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T)
        """
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(np.array(self.rew_buf, dtype=np.float32)[path_slice], last_val)
        vals = np.append(np.array(self.val_buf, dtype=np.float32)[path_slice], last_val)

        self.rtg_buf += discount_cumsum(rews, self.gamma)[:-1].tolist()
        self.adv_buf += gae_lambda(rews, vals, self.gamma).tolist()

        self.path_start_idx = self.ptr

    def get(self) -> PPOBufferItem:
        data = PPOBufferItem(self.device, self.obs_buf, self.act_buf, self.logp_buf,
                             self.rtg_buf, self.adv_buf)
        self._set_buffers()
        return data

    def _add_transition(self, obs, act, rew, val, logp):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)

    def _set_buffers(self):
        self.ptr, self.path_start_idx = 0, 0
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.rtg_buf = []
        self.adv_buf = []


class PPOLagBuffer(PPOBuffer):
    def __init__(self, device: torch.device, gamma: float = 0.99) -> None:
        self.device = device
        self.gamma = gamma
        self._set_buffers()

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float,
            cost: np.ndarray, val_cost: np.ndarray):
        self._add_transition(obs, act, rew, val, logp, cost, val_cost)
        self.ptr += 1

    def path_done(self, last_val: float, last_val_cost: float):
        """
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T)
        """
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(np.array(self.rew_buf, dtype=np.float32)[path_slice], last_val)
        vals = np.append(np.array(self.val_buf, dtype=np.float32)[path_slice], last_val)
        costs = np.append(np.array(self.cost_buf, dtype=np.float32)[path_slice], last_val_cost)
        vals_cost = np.append(np.array(self.val_cost_buf, dtype=np.float32)[path_slice], last_val_cost)

        self.rtg_buf += discount_cumsum(rews, self.gamma)[:-1].tolist()
        self.adv_buf += gae_lambda(rews, vals, self.gamma).tolist()
        self.ctg_buf += discount_cumsum(costs, self.gamma)[:-1].tolist()
        self.adv_cost_buf += gae_lambda(costs, vals_cost, self.gamma).tolist()

        self.path_start_idx = self.ptr

    def get(self) -> PPOLagBufferItem:
        data = PPOLagBufferItem(self.device, self.obs_buf, self.act_buf, self.logp_buf,
                                self.rtg_buf, self.adv_buf, self.cost_buf, self.adv_cost_buf)
        self._set_buffers()
        return data

    def _add_transition(self, obs, act, rew, val, logp, cost, val_cost):
        super()._add_transition(obs, act, rew, val, logp)
        self.cost_buf.append(cost)
        self.val_cost_buf.append(val_cost)

    def _set_buffers(self):
        super()._set_buffers()
        self.cost_buf = []
        self.val_cost_buf = []
        self.ctg_buf = []
        self.adv_cost_buf = []


class PPO:
    def __init__(self, env, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on {self.device}')
        self.env = env
        self.env_name = env.__class__.__name__
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        self._init_hyperparameters(**kwargs)
        self._init_seed()
        self._init_networks()
        self.buffer = PPOBuffer(self.device, self.gam)
        self.logger = Logger()
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPO_' + self.env_name + self.time_str)

    def learn(self):
        self.num_eps = 0
        step = 0
        with tqdm(total=self.total_steps) as pbar:
            while step < self.total_steps:
                # get episodes
                data, epoch_steps = self.rollout()
                # update networks
                self.update_policy(data)
                self.update_value_func(data)
                # record the number of steps in the epoch
                step += epoch_steps
                pbar.update(epoch_steps)

    def rollout(self) -> Union[PPOBufferItem, int]:
        epoch_step = 0
        while epoch_step < self.min_epoch_steps:
            obs, info = self.env.reset(seed=self.seed)
            eps_ret, eps_cost, eps_len = 0, 0, 0  # episode record
            while True:
                act, logp, _ = self.select_action(obs)
                obs_next, rew, term, trun, info = self.env.step(act)
                self.buffer.add(obs, act, rew, self.evaluate(obs), logp)

                obs = obs_next
                eps_ret, eps_cost, eps_len = eps_ret + rew, eps_cost + info["cost"], eps_len + 1

                if term or trun:
                    last_val = 0 if term else self.evaluate(obs)
                    self.buffer.path_done(last_val)

                    epoch_step += eps_len
                    self.num_eps += 1
                    eps_info = {'eps_ret': eps_ret, 'eps_cost': eps_cost, 'eps_len': eps_len}
                    self.logger.add(eps_info)
                    if self.use_tb:
                        self._to_tb(eps_info)
                    break
        return self.buffer.get(), epoch_step

    def select_action(self, obs: np.ndarray) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            act, logp, mean = self.policy(obs2ten(obs, self.device))
        return ten2arr(act), ten2arr(logp), ten2arr(mean)

    def evaluate(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            v_obs = self.value(obs2ten(obs, self.device))
        return ten2arr(v_obs)[0]

    def update_policy(self, data: PPOBufferItem):
        adv = data.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        for _ in range(self.num_updates):
            log_probs = self.policy(data.observations, data.actions).squeeze()
            ratio = torch.exp(log_probs - data.log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_policy = (-torch.min(surr1, surr2)).mean()

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()

    def update_value_func(self, data: PPOBufferItem):
        for _ in range(self.num_updates):
            loss_val = ((self.value(data.observations).squeeze() - data.reward_to_gos) ** 2).mean()
            self.value_optimizer.zero_grad()
            loss_val.backward()
            self.value_optimizer.step()

    def _init_hyperparameters(self, **kwargs):
        self.seed = kwargs.get('seed', 0)
        self.min_epoch_steps = kwargs.get('min_epoch_steps', 1000)
        self.total_steps = kwargs.get('total_steps', 300000)
        self.clip_ratio = kwargs.get('clip_ratio', 0.2)
        self.num_updates = kwargs.get('num_updates', 5)
        self.gam = kwargs.get('gamma', 0.95)
        self.policy_lr = kwargs.get('policy_lr', 0.005)
        self.value_lr = kwargs.get('v_lr', 0.005)
        self.use_tb = kwargs.get('use_tb', False)

    def _init_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _init_networks(self):
        self.policy = GaussianPolicy(self.env).to(self.device)
        self.value = Value(self.env).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=self.value_lr)

    def _to_tb(self, eps_info):
        for name, scalar in eps_info.items():
            self.writer.add_scalar(f'charts/{name}', scalar, self.num_eps)

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))



class PPOLag(PPO):
    def __init__(self, env, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on {self.device}')
        self.env = env
        self.env_name = env.__class__.__name__
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        self._init_hyperparameters(**kwargs)
        self._init_seed()
        self._init_networks()
        self.buffer = PPOLagBuffer(self.device, self.gam)
        self.logger = Logger()
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPOLag_' + self.env_name + self.time_str)

    def rollout(self) -> Union[PPOLagBufferItem, int]:
        epoch_step = 0
        while epoch_step < self.min_epoch_steps:
            obs, info = self.env.reset(seed=self.seed)
            eps_ret, eps_cost, eps_len = 0, 0, 0  # episode record
            while True:
                act, logp, _ = self.select_action(obs)
                obs_next, rew, term, trun, info = self.env.step(act)

                cost = info["cost"]
                val, val_cost = self.evaluate(obs)
                self.buffer.add(obs, act, rew, val, logp, cost, val_cost)

                obs = obs_next
                eps_ret, eps_cost, eps_len = eps_ret + rew, eps_cost + cost, eps_len + 1

                if term or trun:
                    if term:
                        last_val, last_val_cost = 0.0, 0.0
                    else:
                        last_val, last_val_cost = self.evaluate(obs)
                    self.buffer.path_done(last_val, last_val_cost)

                    epoch_step += eps_len
                    self.num_eps += 1
                    eps_info = {'eps_ret': eps_ret, 'eps_cost': eps_cost, 'eps_len': eps_len}
                    self.logger.add(eps_info)
                    if self.use_tb:
                        self._to_tb(eps_info)
                    break
        return self.buffer.get(), epoch_step

    def evaluate(self, obs: np.ndarray) -> Union[float, float]:
        with torch.no_grad():
            obs_tensor = obs2ten(obs, self.device)
            v_obs = self.value(obs_tensor)
            v_cost_obs = self.value_cost(obs_tensor)
        return ten2arr(v_obs)[0], ten2arr(v_cost_obs)[0]

    def update_policy(self, data: PPOLagBufferItem):
        # update penalty parameter
        cur_cost = self.logger.mean('eps_costs')
        loss_penalty = -self.penalty_param * (cur_cost - self.cost_limit)
        self.penalty_optimizer.zero_grad()
        loss_penalty.backward()
        self.penalty_optimizer.step()

        adv = data.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        adv_cost = data.advantages_cost
        adv_cost = (adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-10)
        for _ in range(self.num_updates):
            log_probs = self.policy(data.observations, data.actions).squeeze()
            ratio = torch.exp(log_probs - data.log_probs)
            # policy loss: reward
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_policy_rew = (-torch.min(surr1, surr2)).mean()
            # policy loss: cost
            loss_policy_cost = (ratio * adv_cost).mean()
            # full policy loss
            p = softplus(self.penalty_param).item()
            loss_policy = 1 / (1 + p) * (loss_policy_rew + p * loss_policy_cost)

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()

    def update_value_func(self, data: PPOLagBufferItem):
        for _ in range(self.num_updates):
            loss_val = ((self.value(data.observations).squeeze() - data.reward_to_gos) ** 2).mean()
            self.value_optimizer.zero_grad()
            loss_val.backward()
            self.value_optimizer.step()

            loss_val_cost = ((self.value_cost(data.observations).squeeze() - data.cost_to_gos) ** 2).mean()
            self.value_cost_optimizer.zero_grad()
            loss_val_cost.backward()
            self.value_cost_optimizer.step()

    def _init_hyperparameters(self, **kwargs):
        super()._init_hyperparameters(**kwargs)
        self.penalty_lr = kwargs.get('penalty_lr', 5e-2)
        self.cost_limit = kwargs.get('cost_limit', 5)

    def _init_networks(self):
        super()._init_networks()
        self.value_cost = Value(self.env).to(self.device)
        self.penalty_param = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=self.device)

        self.value_cost_optimizer = Adam(self.value_cost.parameters(), lr=self.value_lr)
        self.penalty_optimizer = Adam([self.penalty_param], lr=self.penalty_lr)