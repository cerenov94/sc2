from stable_baselines3 import PPO
import os
from SC2ENV import Sc2Env
import time
from wandb.integration.sb3 import WandbCallback
import wandb

model_name = f"{int(time.time())}"

models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = Sc2Env()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
if __name__ == '__main__':
    iters = 0
    while True:
        print("On iteration: ", iters)
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
