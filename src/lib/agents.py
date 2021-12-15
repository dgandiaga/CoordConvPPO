import numpy as np
import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from lib.model_definitions.models import CoordConvNet, Net


class Agent:

    # Agent parameters
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 64
    gamma = 0.99

    def __init__(self, model, img_stack=4):
        self.training_step = 0

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Policy net architecture
        if model == 'coordconvnet':
            img_size = 48
            self.net = CoordConvNet(img_stack).double().to(self.device)
        elif model == 'convnet':
            img_size = 96
            self.net = Net(img_stack).double().to(self.device)

        # Memory structure
        transition = np.dtype([('s', np.float64, (img_stack, img_size, img_size)), ('a', np.float64, (3,)),
                               ('a_logp', np.float64), ('r', np.float64),
                               ('s_', np.float64, (img_stack, img_size, img_size))])
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        # Net optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state, deterministic=False):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)

        # Choose among deterministic or stochastic policy
        if deterministic:
            action = alpha / (alpha + beta)
        else:
            action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, model_name):
        torch.save(self.net.state_dict(), f'models/{model_name}.pt')

    def load_param(self, model_name):
        self.net.load_state_dict(torch.load(f'models/{model_name}.pt'))

    def store(self, transition):
        # Store transitions and return if it's time to train the policy net
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            # We calculate the target state values prior to the training loop, and use their differences with the state
            # values of the original state as advantages
            target_v = r + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]

        for _ in tqdm.tqdm(range(self.ppo_epoch), total=self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                # In every batch we get the beta distribution with the current network, and calculate the exponential
                # ratio between the action log probability of the action chosen back then and now
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                # The action loss is the negative mean of the product between ratio and advantage. We use two different
                # versions, of the potential update, one with the ratio as it was calculated and the second one with it
                # restricted to [0.9, 1.1] and we take the lowest option for each action, so we make sure that ratios
                # don't go too high making the training unstable
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                # The value loss is just the smooth L1 loss between the current state values returned by the net and
                # the target state values we calculated before
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])

                # And the final loss is the sum of both of them, giving double weight to the state loss
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
