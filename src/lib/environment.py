import numpy as np
import gym


class Env:

    def __init__(self, model, img_stack=4, action_repeat=8):
        self.env = gym.make('CarRacing-v0')
        if model == 'coordconvnet':
            self.img_size = 48
        elif model == 'convnet':
            self.img_size = 96

        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        img_rescaled = self.rescale(img_gray, self.img_size)

        self.stack = [img_rescaled] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        img_rescaled = self.rescale(img_gray, self.img_size)
        self.stack.pop(0)
        self.stack.append(img_rescaled)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rescale(grayscale_image, img_size):
        return grayscale_image.reshape((img_size, 96 // img_size, img_size, 96 // img_size)).max(3).max(1)

    @staticmethod
    def rgb2gray(rgb, norm=True):

        # RGB image to normalized grayscale
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
