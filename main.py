import gymnasium as gym
import imageio

from mad_pod_racing import MapPodRacing
# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi():
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing,
        max_episode_steps=500,  # Prevent infinite episodes
    )
    gym.pprint_registry()
    env = gym.make("gymnasium_env/MapPodRacing-v0")
    frames = []
    env.reset()
    frames.append(env.render())
    env.close()
    imageio.mimsave("cartpole_episode.gif", frames, fps=1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
