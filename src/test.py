import argparse
import numpy as np

from environment import Env
from agents import Agent

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--model', required=True, help='name of the model stored in models folder')
args = parser.parse_args()

if __name__ == "__main__":
    agent = Agent()
    agent.load_param(args.model)
    env = Env()

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(100):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, _ = agent.select_action(state, deterministic=True)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
