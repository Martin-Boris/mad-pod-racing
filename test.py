import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from simple_dqn_torch_rl import Agent
import torch as T

import imageio

from mad_pod_racing import MapPodRacing



if __name__ == '__main__':
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing,
        max_episode_steps=500,  # Prevent infinite episodes
    )
    env = gym.make("gymnasium_env/MapPodRacing-v0")
    frames = []
    observation, info = env.reset()

    done = False
    while not done:
        action = 1#env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        done = terminated or truncated

    imageio.mimsave("mad_pod_episode.gif", frames, fps=10)