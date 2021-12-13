import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import os

from environment import Env
from agents import Agent

def compare_models():

    model_names = [f.split('_')[0] for f in os.listdir('results/individual') if f.endswith('.csv')]
    model_names = set(model_names)
    print(model_names)
    plt.figure()
    for f in [f for f in os.listdir('results/individual') if f.endswith('.csv')]:

        results = pd.read_csv(f'results/individual/{f}')
        plt.plot(results['avg_reward'], label=f.split('.')[0])
    plt.title('Rewards over episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'results/comparison/compared_models.png')

def compare_models_time():

    model_names = [f.split('_')[0] for f in os.listdir('results/individual') if f.endswith('.csv')]
    model_names = set(model_names)
    print(model_names)
    plt.figure()
    for f in [f for f in os.listdir('results/individual') if f.endswith('.csv')]:

        results = pd.read_csv(f'results/individual/{f}')
        plt.plot(results['time'], results['avg_reward'], label=f.split('.')[0])
    plt.title('Rewards over time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'results/comparison/compared_models_time.png')


def generate_gif(model_name):
    agent = Agent()
    agent.load_param(model_name)
    env = Env()
    img_array = []

    for i_ep in range(1):
        state = env.reset()
        first_image = Image.fromarray(env.env.render(mode="rgb_array"))

        for t in range(1000):
            action, _ = agent.select_action(state, deterministic=True)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            img_array.append(Image.fromarray(env.env.render(mode="rgb_array")))

            state = state_
            if done or die:
                first_image.save("results/samples/out.gif", save_all=True, append_images=img_array, duration=100, loop=0)
                break


if __name__ == "__main__":
    compare_models()
    compare_models_time()
    generate_gif('coordconv_2021-12-13_11:47:37')
    print('FINISHED')