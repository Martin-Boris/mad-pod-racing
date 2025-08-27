import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from simple_dqn_torch_rl import Agent
import torch as T

import imageio

from mad_pod_racing import MapPodRacing


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
    gym.register(
        id="gymnasium_env/MapPodRacing-v0",
        entry_point=MapPodRacing,
        max_episode_steps=500,  # Prevent infinite episodes
    )
    env = gym.make("gymnasium_env/MapPodRacing-v0")
    agent = Agent(gamma=0.99,epsilon=1.0,batch_size=500,n_actions=12,eps_end=0.01,input_dims=[8],lr=0.003)
    scores =[]
    eps_history= []
    cp_dones = []
    n_games = 200

    for i in range(n_games):
        score = 0
        cp_completion = 0
        done = False
        observation, info = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            cp_completion = info["cp_completion"]
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        cp_dones.append(cp_completion)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_cp_done = np.mean(cp_dones[-100:])

        print('episode ', i, ' score %.2f' % score, 'avg score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon, 'avg cp %.2f' % avg_cp_done)

    x= [i+1 for i in range(n_games)]
    filename = "mad_pod_racing.png"
    plot_learning_curve(x, scores, eps_history, filename)

    weight,bias = agent.extract_parameter()
    print("weight")
    print(weight[0].shape)
    print(weight[1].shape)
    print(weight[2].shape)
    print(weight[3].shape)
    print("bias")
    print(bias[0].shape)
    print(bias[1].shape)
    print(bias[2].shape)
    print(bias[3].shape)

    score = 0
    done = False
    observation, info = env.reset()
    frames = [env.render()]
    with T.no_grad():
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            frames.append(env.render())
            observation = observation_
    print("score ",str(score))
    imageio.mimsave("mad_pod_episode.gif", frames, fps=10)






