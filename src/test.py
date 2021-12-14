import argparse
import numpy as np
import time

from lib.environment import Env
from lib.agents import Agent

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--episodes', default=100, type=int, help='render the environment')
parser.add_argument('--model-name', required=True, help='name of the model stored in models folder')
parser.add_argument('--sleep-time', default=0, type=float,
                    help='time to sleep between actions in order to speed down visualization')
args = parser.parse_args()

if __name__ == "__main__":
    agent = Agent(args.model_name.split('_')[0])
    agent.load_param(args.model_name)
    env = Env(args.model_name.split('_')[0])

    running_score = []
    state = env.reset()
    for i_ep in range(args.episodes):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, _ = agent.select_action(state, deterministic=True)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                time.sleep(args.sleep_time)
                env.render()
            score += reward
            state = state_
            if done or die:
                break
        running_score.append(score)
        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    print(f'Average reward per episode: {np.mean(running_score)}')
