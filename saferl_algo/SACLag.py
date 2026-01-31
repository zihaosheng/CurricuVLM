"""
https://github.com/ammarhydr/SAC-Lagrangian/blob/main/SAC_Agent.py
https://github.com/ammarhydr/SAC-Lagrangian/issues/3
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .SAC import soft_update, hard_update
from .SAC import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
class SACLag(object):
    def __init__(self, num_inputs, action_space, args, lambda_init=1.):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.lab_update_interval = 12
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        self.critic_c = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim_c = Adam(self.critic_c.parameters(), lr=args.lr)
        self.critic_target_c = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        hard_update(self.critic_target_c, self.critic_c)

        self.lam = torch.tensor(lambda_init, requires_grad=True)
        self.lam_optimiser = torch.optim.Adam([self.lam], lr=3e-4)

        self.cost_lim = 5

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        cost_batch = torch.FloatTensor(cost_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # calculating the target q
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

            # calculating the target q_cost
            qf1_next_target_cost, qf2_next_target_cost = self.critic_target_c(next_state_batch, next_state_action)
            min_qf_next_target_cost = torch.min(qf1_next_target_cost, qf2_next_target_cost) - self.alpha * next_state_log_pi
            next_q_value_cost = cost_batch + mask_batch * self.gamma * (min_qf_next_target_cost)

        # calculating q
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # calculating q_cost
        qf1_cost, qf2_cost = self.critic_c(state_batch,
                                           action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss_cost = F.mse_loss(qf1_cost,
                                   next_q_value_cost)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss_cost = F.mse_loss(qf2_cost,
                                   next_q_value_cost)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss_cost = qf1_loss_cost + qf2_loss_cost
        self.critic_optim_c.zero_grad()
        qf_loss_cost.backward()
        self.critic_optim_c.step()

        # updating the policy gradient ascent
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # updating the cost gradient decent
        qf1_pi_cost, qf2_pi_cost = self.critic_c(state_batch, pi)
        min_qf_pi_cost = torch.min(qf1_pi_cost, qf2_pi_cost)
        penalty = self.lam * (min_qf_pi_cost - self.cost_lim)
        policy_loss = (inside_term + penalty).sum(dim=1).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.lab_update_interval == 0:
            violation = self.cost_lim - min_qf_pi_cost
            self.log_lam = torch.nn.functional.softplus(self.lam)
            lambda_loss = self.log_lam * violation.detach()
            lambda_loss = lambda_loss.sum()
            lambda_loss.backward(torch.ones_like(lambda_loss))
            self.lam_optimiser.step()

        if self.automatic_entropy_tuning:

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.critic_target_c, self.critic_c, self.tau)

        return qf1_loss.item(), qf2_loss.item(), qf1_loss_cost.item(), qf2_loss_cost.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), self.log_lam.item()

        # Save model parameters

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

        # Load model parameters

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.critic_c.eval()
                self.critic_target_c.eval()

            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                self.critic_c.train()
                self.critic_target_c.train()


"""
This code is adopted from https://github.com/pranz24/pytorch-soft-actor-critic
"""
import random
import numpy as np
from operator import itemgetter

class ReplayMemoryLag:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    # state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch
    def push(self, state, action, reward, cost, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, cost, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, cost, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, cost, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)  # 如果len(self.buffer) < batch_size，也会重复取到batch_size
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, cost, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, cost, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)