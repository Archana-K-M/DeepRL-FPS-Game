from stable_baselines3 import PPO
from deathm import VizDoomDeathmatchEnv

env = VizDoomDeathmatchEnv(render=True)
model = PPO.load("ppo_deathmatch_finetuned")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()