import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

from lib.environment import Env
from lib.agents import Agent


parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--model', type=str, choices=['convnet', 'coordconvnet'], default='coorconvnet',
                    help='model architecture')
args = parser.parse_args()


if __name__ == "__main__":

    # Agent/Environment initialization
    agent = Agent(args.model)
    env = Env(args.model)

    # Variables for logging purposes and metrics
    date = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    model_name = f'{args.model}_{date}'
    n_params = sum(p.numel() for p in agent.net.parameters())
    starting_time = time.time()
    training_records = []
    episode_rewards = []
    accumulated_reward = []
    results = pd.DataFrame(columns=['model', 'date', 'episode', 'time', 'reward', 'avg_reward', 'n_params'])
    best_acc_reward = -100

    # 2000 epochs set as experimentation set-up
    for i_ep in range(2000):
        episode_rewards.append(0)
        state = env.reset()

        # Episode initialization
        for t in range(1000):

            # Action selection
            action, a_logp = agent.select_action(state)

            # Environment step
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

            # Render
            if args.render:
                env.render()

            # Store and update the model if the batch limit is reached
            if agent.store((state, action, a_logp, reward, state_)):
                print('Updating policy...')
                agent.update()

            # Store reward and update state
            episode_rewards[-1] += reward
            state = state_
            if done or die:
                break

        # Store accumulated rewards and results, and save them to disk
        if len(episode_rewards) < 100:
            accumulated_reward.append(np.mean(episode_rewards))
        else:
            accumulated_reward.append(np.mean(episode_rewards[-100:]))

        results.loc[len(results)] = [args.model, date, i_ep, time.time()-starting_time, episode_rewards[-1],
                                     accumulated_reward[-1], n_params]
        results.to_csv(f'results/individual/{model_name}.csv', index=False)

        # Updating visualization and saving model to disk every 10 episodes
        if i_ep % 10 == 0:
            plt.figure()
            plt.plot(episode_rewards, label='rewards')
            plt.plot(accumulated_reward, label='averaged_reward_over_100_runs')
            plt.title('Rewards over time')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.savefig(f'results/individual/{model_name}.png')
            plt.close()
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, episode_rewards[-1],
                                                                                   accumulated_reward[-1]))

            if best_acc_reward < accumulated_reward[-1]:
                agent.save_param(model_name)
                best_acc_reward = accumulated_reward[-1]
                print('Saving model into disk...')
