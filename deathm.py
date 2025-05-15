import os
import numpy as np
import gymnasium as gym
from vizdoom import DoomGame, Mode, ScreenResolution
from gymnasium import spaces
import cv2

class VizDoomDeathmatchEnv(gym.Env):
    def __init__(self, render=True):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(r"C:\Users\admin\Desktop\ml\scenarios\deathmatch.cfg")
        self.game.set_doom_scenario_path(r"C:\Users\admin\Desktop\ml\scenarios\deathmatch.wad")

        self.game.set_window_visible(render)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()

        self.actions = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
            [0, 0, 0] 
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(480, 640, 3), dtype=np.uint8)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            obs = self.game.get_state().screen_buffer
            obs = np.transpose(obs, (1, 2, 0))

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = np.transpose(state, (1, 2, 0))
        return state, {}

    def render(self):
        pass

    def close(self):
        self.game.close()

# ------------------------
# Enemy Detection Heuristic
# ------------------------
def detect_enemy_direction(frame):
    """Detects enemy in frame and returns action:
       1 (left), 2 (right), 3 (shoot), or 0 (move forward)"""

    # Resize + Blur to simplify
    img = cv2.resize(frame, (160, 120))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Enemy colors range (brown/red in HSV)
    lower_enemy = np.array([0, 100, 50])
    upper_enemy = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_enemy, upper_enemy)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return 0  # fallback

        cx = int(M["m10"] / M["m00"])

        if cx < 50:
            return 1  # turn left
        elif cx > 110:
            return 2  # turn right
        else:
            return 3  # shoot
    else:
        return 0  # move forward

# ------------------------
# Run Agent with Aiming
# ------------------------
if __name__ == "__main__":
    env = VizDoomDeathmatchEnv(render=True)

    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = detect_enemy_direction(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            if done:
                print(f"Episode {episode + 1} total reward: {total_reward}")
                break

    env.close()
