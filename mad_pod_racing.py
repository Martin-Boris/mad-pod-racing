import random
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
import uuid
import pygame
import math

from gymnasium import spaces
from classes import Map, Vector, POD_RADIUS, CHECKPOINT_RADIUS, point_to_segment_distance, from_vector


"""
Case sigmoid action space 9 discrete :
    - -1 for nothing
    - 0 rad for angle 0 or 360° or 2PI for angle 360
    - 45° or PI/4 for angle 45
    - 90° or PI/2 for angle 90
    - 135° or 3PI/4 for angle 135
    - 180° or PI for angle 180
    - 225° or 5PI/4 for angle 225
    - 270° or 3PI/2 for angle 270
    - 315° or 7PI/3 for angle 315
    
Case Relu :
    - 1 Box between 0 and 2PI
         
    
observation space : 
   - position (x,y)
   - next checkpoint position (x,y)
   - next checkpoint distance (one value d)
   - next checkpoint angle (one value in rad)
   - speed (vector x,y)
   
rewards:
    - -1 per turn without checkpoint
    - +20 when end
    - +5 when checkpoint

end :
    - lap over : i.e all checkpoint validated in right order 3 times
    - 100 round without reaching a checkpoint
"""
ENV_WIDTH = 16000
ENV_HEIGHT = 9000
MAX_SPEED = 15000
TIME_OUT = 100
CP_REWARD = 1
END_REWARD = 20
TRAVEL_REWARD = -0.01
OUT_SCREEN_REWARD = 0#-100

class MapPodRacing(gym.Env):

    def __init__(self):
        self.cp_queue = None
        self.timeout = None
        self.my_pod = None
        self.trajectory_reward = None
        self.map = None
        self.seed = None
        self.cp_done = 0
        self.action_space = gym.spaces.Discrete(12)
        '''self.angle_map = np.array([
            0,  # 0°
            np.pi / 4,  # 45°
            np.pi / 2,  # 90°
            3 * np.pi / 4,  # 135°
            np.pi,  # 180°
            5 * np.pi / 4,  # 225°
            3 * np.pi / 2,  # 270°
            7 * np.pi / 4  # 315°
        ])'''

        self.angle_map = np.array([
            0,  # 0°
            np.pi / 6,  # 30°
            np.pi / 3,  # 60°
            np.pi / 2,  # 90°
            2 * np.pi / 3,  # 120°
            5 * np.pi / 6,  # 150°
            np.pi,  # 180°
            7 * np.pi / 6,  # 210°
            4 * np.pi / 3,  # 240°
            3 * np.pi / 2,  # 270°
            5 * np.pi / 3,  # 300°
            11 * np.pi / 6  # 330°
        ])

        """self.action_space = spaces.Box(
            low=0.0,
            high=2 * np.pi,
            shape=(1,),
            dtype=np.float32
        )"""

        # Observation space (Box of 8 values)
        low = np.array([
            -2000, -2000,  # position x, y
            0, 0,  # checkpoint x, y
            0.0,  # distance (>= 0)
            -np.pi*2,  # angle (in radians)
            -MAX_SPEED, -MAX_SPEED  # speed x, y
        ], dtype=np.float32)

        high = np.array([
            ENV_WIDTH+2000, ENV_HEIGHT+2000,  # position x, y
            ENV_WIDTH, ENV_HEIGHT,  # checkpoint x, y
            ENV_WIDTH*2,  # distance
            np.pi*2,  # angle
            MAX_SPEED, MAX_SPEED  # speed x, y
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # render
        self.image_ratio = 25
        self.image_width= ENV_WIDTH / self.image_ratio
        self.image_heigh= ENV_HEIGHT / self.image_ratio


    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.seed = uuid.uuid4().int & ((1 << 64) - 1)
        random.seed(self.seed)
        self.map = Map(self.seed)
        self.trajectory_reward = 0
        # Player information
        self.my_pod = random.choice(self.map.pods)
        self.timeout = TIME_OUT
        self.cp_done = 0
        self.cp_queue = deque(maxlen=18)
        for _ in range(3):
            self.cp_queue.extend(self.map.check_points)
        last_cp = self.cp_queue.popleft()
        self.cp_queue.append(last_cp)
        observation = self.get_obs()
        return observation, { 'cp': self.cp_queue}

    def get_obs(self):
        cp_x, cp_y = 0,0
        if len(self.cp_queue) > 0:
            cp_x, cp_y = self.cp_queue[0]
        distance = math.sqrt((self.my_pod.position.x - cp_x) ** 2 + (self.my_pod.position.y - cp_y) ** 2)
        return np.array([self.my_pod.position.x, self.my_pod.position.y,
                  cp_x, cp_y,
                  distance,
                  from_vector(self.my_pod.position, Vector(cp_x, cp_y)).angle(),
                  self.my_pod.speed.x, self.my_pod.speed.y
                  ], dtype=np.float32)

    def step(self, action):
        # apply action on my pod
        angle = self.angle_map[action]
        self.my_pod.update_acceleration_from_angle(angle,100)
        self.my_pod.apply_force(self.my_pod.acceleration)
        self.my_pod.step()
        self.my_pod.apply_friction()
        self.my_pod.end_round()
        terminated = False
        truncated = False
        reward =0

        if point_to_segment_distance( self.cp_queue[0][0],  self.cp_queue[0][1],
                                     self.my_pod.last_position.x, self.my_pod.last_position.y,
                                    self.my_pod.position.x,   self.my_pod.position.y) <= CHECKPOINT_RADIUS:
            self.cp_queue.popleft()
            self.timeout = TIME_OUT
            if len(self.cp_queue) ==0:
                reward += END_REWARD
                terminated = True
            else:
                #reward = CP_REWARD
                self.cp_done +=1
        else:
            reward += TRAVEL_REWARD
            self.timeout -= 1
            if self.timeout <= 0:
                terminated = True
        self.trajectory_reward += reward
        obs = self.get_obs()
        info = {"cp_completion": 1-(len(self.cp_queue)/ (len(self.map.check_points)*3))}
        return obs, reward, terminated, truncated, info

    def render(self):
        canvas = pygame.Surface((self.image_width, self.image_heigh))
        canvas.fill((255, 255, 255))
        pygame.font.init()
        red = (255, 0, 0)
        past_red = (255, 100, 0)
        blue = (0, 0, 255)
        dark_grey = (64, 64, 64)
        font = pygame.font.Font(None, 36)
        text = font.render(str(self.trajectory_reward), True, blue)
        canvas.blit(text, (10,10))

        cp_order = 0
        for checkpoint in self.map.check_points:
            pygame.draw.circle(canvas, dark_grey, np.array(checkpoint) / self.image_ratio,
                               CHECKPOINT_RADIUS / self.image_ratio)
            cp_text = font.render(str(cp_order), True, blue)
            canvas.blit(cp_text, (checkpoint[0] / self.image_ratio, checkpoint[1] / self.image_ratio))
            cp_order +=1
        pygame.draw.circle(canvas, past_red, np.array(self.my_pod.last_position.get_tuple()) / self.image_ratio,
                           POD_RADIUS / self.image_ratio)
        pygame.draw.circle(canvas, red, np.array(self.my_pod.position.get_tuple()) / self.image_ratio,
                               POD_RADIUS / self.image_ratio)
        pygame.draw.line(canvas, blue, np.array(self.my_pod.last_position.get_tuple()) / self.image_ratio,
                         np.array(self.my_pod.position.get_tuple()) / self.image_ratio, 1)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )


    """def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()"""

def supervised_action_choose(observation):
    target_angle = observation[5]  # angle to cp (in radians)
    angles = np.array([
        0,  # 0°
        np.pi / 6,  # 30°
        np.pi / 3,  # 60°
        np.pi / 2,  # 90°
        2 * np.pi / 3,  # 120°
        5 * np.pi / 6,  # 150°
        np.pi,  # 180°
        7 * np.pi / 6,  # 210°
        4 * np.pi / 3,  # 240°
        3 * np.pi / 2,  # 270°
        5 * np.pi / 3,  # 300°
        11 * np.pi / 6  # 330°
    ])
    # Normalize angular differences to [-π, π]
    angle_diffs = np.abs((angles - target_angle + np.pi) % (2 * np.pi) - np.pi)

    action = np.argmin(angle_diffs)
    return action