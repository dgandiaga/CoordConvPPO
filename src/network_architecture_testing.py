import unittest

import torch
from lib.environment import Env
from lib.agents import Agent


class TestDatasets(unittest.TestCase):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def test_convnet_environment_compatibility(self):

        env = Env('convnet')

        state = env.reset()
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)

        agent = Agent('convnet')

        (alpha, beta), v = agent.net(state)

        print(f'Agent-Environment network compatibility checked for model convnet')

    def test_coordconvnet_environment_compatibility(self):

        env = Env('coordconvnet')

        state = env.reset()
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)

        agent = Agent('coordconvnet')

        (alpha, beta), v = agent.net(state)

        print(f'Agent-Environment network compatibility checked for model coordconvnet')


if __name__ == '__main__':
    unittest.main()
