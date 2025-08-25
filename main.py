import gymnasium as gym
import imageio

from mad_pod_racing import MapPodRacing
# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def simulate_one_game():
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
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        done = terminated or truncated

    imageio.mimsave("mad_pod_episode.gif", frames, fps=10)

    for i, frame in enumerate(frames):
        filename = f"test_frames/frame_{i:04d}.png"  # Zero-padded numbering
        imageio.imwrite(filename, frame)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    simulate_one_game()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
