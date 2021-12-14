import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import scipy.stats as ss
import torch
import os
import argparse

from lib.environment import Env
from lib.agents import Agent

parser = argparse.ArgumentParser(description='Visualization generator and model comparison')
parser.add_argument('--model-name', default='', help='name of the model stored in models folder')
args = parser.parse_args()


def compare_models():
    model_names = [f for f in os.listdir('results/individual') if f.endswith('.csv')]
    for f in model_names:
        try:
            results = results.append(pd.read_csv(f'results/individual/{f}'))
        except NameError:
            results = pd.read_csv(f'results/individual/{f}')

    plt.figure()
    for model in results['model'].unique():

        results_data = results[results['model'] == model]
        num_of_samples = results_data['date'].nunique()
        results_data_grouped = results_data.groupby(['episode']).agg({'avg_reward': ('mean', 'std')})
        label = f'{model} - nº of runs: {num_of_samples}'
        results_reduced = results_data_grouped[(results_data_grouped.index % 50) == 0]

        plt.errorbar(results_reduced.index, results_reduced['avg_reward']['mean'],
                     yerr=results_reduced['avg_reward']['std'], label=label, fmt='--o', capsize=6)
    plt.legend()
    plt.title('Avg reward over 100 runs per episode')
    plt.xlabel('Episode')
    plt.ylabel('Avg reward')
    plt.savefig(f'results/comparison/compared_models.png')
    plt.close()
    print('Generated model comparison over episode in results/comparison/compared_models.png')


def compare_models_time():
    model_names = [f for f in os.listdir('results/individual') if f.endswith('.csv')]
    for f in model_names:
        try:
            results = results.append(pd.read_csv(f'results/individual/{f}'))
        except NameError:
            results = pd.read_csv(f'results/individual/{f}')

    plt.figure()
    for model in results['model'].unique():

        results_data = results[results['model'] == model]
        num_of_samples = results_data['date'].nunique()
        results_data_grouped = results_data.groupby(['episode']).agg({'avg_reward': ('mean', 'std'), 'time': 'mean'})
        label = f'{model} - nº of runs: {num_of_samples}'
        results_reduced = results_data_grouped[(results_data_grouped.index % 50) == 0]

        plt.errorbar(results_reduced['time'], results_reduced['avg_reward']['mean'],
                     yerr=results_reduced['avg_reward']['std'], label=label, fmt='--o', capsize=6)
    plt.legend()
    plt.title('Avg reward over 100 runs per time')
    plt.xlabel('Time')
    plt.ylabel('Avg reward')
    plt.savefig(f'results/comparison/compared_models_time.png')
    plt.close()
    print('Generated model comparison over time in results/comparison/compared_models_time.png')


def generate_gif_and_examples(model_name):
    for f in os.listdir('results/samples'):
        os.remove(f'results/samples/{f}')
    model = model_name.split('_')[0]
    agent = Agent(model)
    agent.load_param(model_name)
    env = Env(model)
    img_array = []

    for i_ep in range(3):
        state = env.reset()
        first_image = Image.fromarray(env.env.render(mode="rgb_array"))
        count = 0

        for t in range(1000):
            action, _ = agent.select_action(state, deterministic=True)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            img_array.append(Image.fromarray(env.env.render(mode="rgb_array")))

            if np.random.random() < 1 / 5:
                plot_beta(agent, env, state, count, model_name, i_ep)
                count += 1

            state = state_
            if done or die:
                first_image.save(f"results/samples/{model_name}_episode_{i_ep}.gif", save_all=True, append_images=img_array,
                                 duration=100, loop=0)
                break

    print('Generated gifs and some examples of action distributions in results/samples')


def plot_beta(agent, env, state, count, model_name, i_ep):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    action_names = ['turn', 'gas', 'brake']
    x_beta = np.linspace(0, 1, 5000)

    fig, ax = plt.subplots(2)
    ax[0].imshow(Image.fromarray(env.env.render(mode="rgb_array")))
    ax[0].set_title('State')
    ax[0].axis('off')

    state = torch.from_numpy(state).double().to(device).unsqueeze(0)
    alpha, beta = agent.net(state)[0]
    alpha, beta = alpha.cpu().detach().numpy()[0], beta.cpu().detach().numpy()[0]
    action_distributions = {action_names[i]: [alpha[i], beta[i]] for i in range(3)}

    for action, (alpha, beta) in action_distributions.items():
        y = ss.beta.pdf(x_beta, alpha, beta)
        ax[1].plot(x_beta, y, label=action)
    ax[1].set_title('Probability distributions for actions')
    ax[1].legend()
    fig.savefig(f'results/samples/{model_name}_episode_{i_ep}_sample_{count}.png')
    plt.close()


if __name__ == "__main__":
    compare_models()
    compare_models_time()
    if len(args.model_name) > 0:
        generate_gif_and_examples(args.model_name)
