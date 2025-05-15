
'''from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from deathm import VizDoomDeathmatchEnv  

env = VizDoomDeathmatchEnv(render=False)
check_env(env, warn=True)
policy_kwargs = dict(
    features_extractor_kwargs=dict(features_dim=256)
)

model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_vizdoom_log/", n_steps=1024, batch_size=64)
model.learn(total_timesteps=100_000)
model.save("ppo_vizdoom_deathmatch")

env.close()'''


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from deathm import VizDoomDeathmatchEnv
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

env = DummyVecEnv([lambda: VizDoomDeathmatchEnv(render=False)])
model = PPO.load("ppo_vizdoom_deathmatch", env=env)

model.learn(total_timesteps=100000)
model.save("ppo_deathmatch_finetuned")

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=2,
    hyperparam_mutations={
        "lr": [1e-4, 5e-4, 1e-3],
        "gamma": [0.95, 0.99, 0.999],
        "clip_range": [0.1, 0.2, 0.3],
        "n_steps": [128, 256, 512],
    })

analysis = tune.run(model,
    config={
        "lr": tune.choice([1e-4, 1e-3]),
        "gamma": tune.choice([0.95, 0.99]),
        "clip_range": tune.choice([0.1, 0.2]),
        "n_steps": tune.choice([128, 256])
    },
    scheduler=pbt,
    stop={"training_iteration": 10},
    verbose=1
)