import matplotlib.pyplot as plt
import pandas as pd
import os

def compare_models():

    model_names = [f.split('_')[0] for f in os.listdir('results/individual') if f.endswith('.csv')]
    model_names = set(model_names)
    print(model_names)
    plt.figure()
    for f in [f for f in os.listdir('results/individual') if f.endswith('.csv')]:

        results = pd.read_csv(f'results/individual/{f}')
        plt.plot(results['avg_reward'], label=f.split('.')[0])
    plt.title('Rewards over time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'results/comparison/compared_models.png')


if __name__ == "__main__":
    compare_models()