import gymnasium as gym
import pygame

from mad_pod_racing import MapPodRacing
# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def cartpole():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = env.action_space.sample()  # Random action for now - real agents will be smarter!

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

def print_hi():
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing,
        max_episode_steps=500,  # Prevent infinite episodes
    )
    gym.pprint_registry()
    env = gym.make("gymnasium_env/MapPodRacing-v0", render_mode="human")
    env.reset()

    pygame.init()
    frame = env.render()

    frame_height, frame_width, _ = frame.shape
    screen = pygame.display.set_mode((frame_width, frame_height))
    pygame.display.set_caption("Gymnasium Env in Pygame")

    clock = pygame.time.Clock()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi()
    cartpole()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
