import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from obstacle_tower_env import ObstacleTowerEnv

if __name__ == "__main__":
    config = {'dense-reward': 1}

    # log name
    experiment_name = "ppo_fixed_"
    # log directory
    experiment_logdir = f"project_logs/{experiment_name}"

    # Initialize environment
    # Retro downscales image input
    env = ObstacleTowerEnv(environment_filename="path_to_ObstacleTower.exe",retro=True, realtime_mode=False, config=config, timeout_wait=300)
    # Set static/not randomized environment
    env.seed(1001)

    # Save checkpoint model
    checkpoint_callback = CheckpointCallback(save_freq=200000, save_path='./checkpoints/',
                                         name_prefix="ppo_default")
    
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=experiment_logdir)
    
    model.learn(total_timesteps=5000000, callback=checkpoint_callback)
    
    model.save("ppo_obstacle_tower")
