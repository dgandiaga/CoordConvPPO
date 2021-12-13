import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

from environment import Env
from agents import Agent


parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--tag', type=str, default='', help='tag for saving results')
parser.add_argument('--img-size', type=int, default=96, help='image size')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()


if __name__ == "__main__":
    agent = Agent(args.img_stack, args.img_size, args.gamma, args.seed)
    n_params = sum(p.numel() for p in agent.net.parameters())
    env = Env(args.img_size, args.img_stack, args.action_repeat, args.seed)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    starting_time = time.time()
    training_records = []
    episode_rewards = []
    accumulated_reward = []
    results = pd.DataFrame(columns=['time', 'reward', 'avg_reward', 'n_params'])

    for i_ep in range(2000):
        episode_rewards.append(0)
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            episode_rewards[-1] += reward
            state = state_
            if done or die:
                break
        if len(episode_rewards) < 100:
            accumulated_reward.append(np.mean(episode_rewards))
        else:
            accumulated_reward.append(np.mean(episode_rewards[-100:]))

        results.loc[len(results)] = [time.time()-starting_time, episode_rewards[-1], accumulated_reward[-1], n_params]
        results.to_csv(f'results/individual/{args.tag}_{date}.csv', index=False)

        if i_ep % args.log_interval == 0:
            plt.figure()
            plt.plot(episode_rewards, label='rewards')
            plt.plot(accumulated_reward, label='averaged_reward_over_100_runs')
            plt.title('Rewards over time')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.savefig(f'results/individual/{args.tag}_{date}.png')
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, episode_rewards[-1],
                                                                                   accumulated_reward[-1]))
            agent.save_param(args.tag, date)
