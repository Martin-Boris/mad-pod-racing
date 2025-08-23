from typing import Optional

import gymnasium as gym
import numpy as np
import uuid
import pygame

from gymnasium import spaces
from gymnasium.core import RenderFrame
from classes import Pod,Map,Vector, POD_RADIUS,CHECKPOINT_RADIUS


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
    - 315° or 5PI/3 for angle 315
    
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
MAX_SPEED = 5000

class MapPodRacing(gym.Env):

    def __init__(self):
        #self.action_space = gym.spaces.Discrete(9)
        self.action_space = spaces.Box(
            low=0.0,
            high=2 * np.pi,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space (Box of 8 values)
        low = np.array([
            -2000, -2000,  # position x, y
            0, 0,  # checkpoint x, y
            0.0,  # distance (>= 0)
            -0,  # angle (in radians)
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
        self.seed = uuid.uuid4().int & ((1 << 64) - 1)
        self.map = Map(self.seed)

        # render
        self.image_ratio = 100
        self.image_width= ENV_WIDTH / self.image_ratio
        self.image_heigh= ENV_HEIGHT / self.image_ratio


    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):
        # Return initial observation matching observation_space
        self.seed = uuid.uuid4().int & ((1 << 64) - 1)
        self.map =Map(self.seed)
        return np.zeros(8, dtype=np.float32), {}



    def step(self, action):
        # Dummy implementation
        obs = np.zeros(8, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.image_width, self.image_heigh))
        canvas.fill((255, 255, 255))

        red = (255, 0, 0)
        dark_grey = (64, 64, 64)
        for checkpoint in self.map.check_points:
            pygame.draw.circle(canvas, dark_grey, np.array(checkpoint)/self.image_ratio, CHECKPOINT_RADIUS/self.image_ratio)
        for pod in self.map.pods:
            pygame.draw.circle(canvas, red, np.array(pod.position.get_tuple())/self.image_ratio, POD_RADIUS/self.image_ratio)

        return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    """def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()"""