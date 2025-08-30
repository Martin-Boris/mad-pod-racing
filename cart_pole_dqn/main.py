import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import numpy as np
import torch

from cart_pole_dqn.agent import Agent
from cart_pole_dqn.helper import plot_durations


def plot_learning_curve(x, scores, epsilon, filename):
    fig=plt.figure()
    ax=fig.add_subplot(111,label="1")
    ax2=fig.add_subplot(111,label="2", frame_on=False)

    ax.plot(x,epsilon, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", labelcolor="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    N=len(scores)

    running_avg=np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", labelcolor="C1")

    plt.savefig(filename)


if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )


    # To ensure reproducibility during training, you can fix the random seeds
    # by uncommenting the lines below. This makes the results consistent across
    # runs, which is helpful for debugging or comparing different approaches.
    #
    # That said, allowing randomness can be beneficial in practice, as it lets
    # the model explore different training trajectories.


    # seed = 42
    # random.seed(seed)
    # torch.manual_seed(seed)
    # env.reset(seed=seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 2500
    TAU = 0.005
    LR = 3e-4


    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)
    agent = Agent(eps_end= EPS_END,eps_start=EPS_START,eps_decay=EPS_DECAY, n_observations=n_observations,n_actions=n_actions, device=device, lr=LR)


    steps_done = 0
    episode_durations = []

    print(torch.cuda.is_available())
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state, env,device)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.push_to_memory(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model(device)

            agent.target_net_update()

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()