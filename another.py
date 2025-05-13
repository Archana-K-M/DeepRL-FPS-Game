import os
import random
import numpy as np
import gymnasium as gym
from vizdoom import DoomGame, Mode, ScreenResolution, Button, GameVariable
from gymnasium import spaces

class VizDoomDeathmatchEnv(gym.Env):
    def __init__(self, render=True):
        super(VizDoomDeathmatchEnv, self).__init__()

        self.game = DoomGame()
        self.game.load_config(r"C:\Users\admin\Desktop\ml\scenarios\deathmatch.cfg")
        self.game.set_doom_scenario_path(r"C:\Users\admin\Desktop\ml\scenarios\deathmatch.wad")

        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()

        # Action space: Move forward/back, turn, shoot
        self.actions = [
            [1, 0, 0],  # move forward
            [0, 1, 0],  # turn left
            [0, 2, 0],  # turn right
            [0, 0, 1],  # shoot
            [0, 0, 0]   # do nothing
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space: raw screen pixels
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(480, 640, 3), dtype=np.uint8)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            obs = self.game.get_state().screen_buffer
            obs = np.transpose(obs, (1, 2, 0))  # CHW to HWC

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = np.transpose(state, (1, 2, 0))
        return state, {}

    def render(self):
        pass  # Already rendered by ViZDoom window

    def close(self):
        self.game.close()

# Test environment
if __name__ == "__main__":
    env = VizDoomDeathmatchEnv()

    for episode in range(10):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                print(f"Episode {episode + 1} reward: {total_reward}")
                break

    env.close()
