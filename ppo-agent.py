import gym
import os
import time
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from obstacle_tower_env import ObstacleTowerEnv
from mlagents_envs.exception import UnityCommunicatorStoppedException, UnityWorkerInUseException, UnityTimeOutException

class FrameSkip(gym.Wrapper):
    # https://github.com/compsciencelab/pytorchrl/blob/master/pytorchrl/envs/common.py
    def __init__(self, env, skip=1):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        last_frame = obs
        return last_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def create_env(worker_id=0):
    config = {'dense-reward': 1}
    env =  ObstacleTowerEnv(
        environment_filename="path/ObstacleTower/ObstacleTower.exe",
        retro=True,
        realtime_mode=False,
        config=config,
        worker_id=random.randint(0, 100),
        timeout_wait=300,
        reduced_actions=True
    )
    env = FrameSkip(env=env, skip=2)
    return env


if __name__ == "__main__":
    config = {'dense-reward': 1}
    
    experiment_name = "ppo_8_reduced_actions"
    experiment_logdir = f"project_logs/{experiment_name}"

    env = make_vec_env(lambda: create_env(), n_envs=4)

    checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./checkpoints/',
                                                 name_prefix="ppo_8_reduced_frameskip")
    env = VecFrameStack(env, n_stack=4)


    model_path = "ppo_8_reduced_frameskip.zip"

    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env, tensorboard_log=experiment_logdir, device="cuda")
    else:
        model = PPO("CnnPolicy", env=env, verbose=1, tensorboard_log=experiment_logdir, device="cuda", n_steps=512, learning_rate=0.0001, n_epochs=8, ent_coef=0.001)
    
    for i in range(400):
        try:
            
            model.learn(total_timesteps=2048, callback=checkpoint_callback, reset_num_timesteps=False, tb_log_name="8ReducedSkip")

            model.save("ppo_8_reduced_frameskip")
            print("Saved")

        except (UnityCommunicatorStoppedException, UnityWorkerInUseException, UnityTimeOutException, RuntimeError) as e:
            print("Unity environment crashed. Restarting training.")
            time.sleep(10)
            env.close()
            env = make_vec_env(lambda: create_env(), n_envs=4)
            env = VecFrameStack(env, n_stack=4)

            model.set_env(env=env)
            continue

    env.close()

