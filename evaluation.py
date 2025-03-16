from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation, UnityGymException
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gym
import random

def run_episode(env, model):
    try:
        done = False
        episode_return = 0.0
        obs = env.reset()
        while not done:
            #action = env.action_space.sample()
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_return += reward
        return episode_return
    except UnityGymException as e:
        return None

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
    #config = {'dense-reward': 1, "tower-seed": 20}
    config = {'dense-reward': 1}

    # Change realtime_mode to True if you want to display agent on screen
    env =  ObstacleTowerEnv(
        environment_filename="path/to/obstacle-tower-env/obstacletower_v4.1_windows/ObstacleTower/ObstacleTower.exe",
        retro=True,
        realtime_mode=False,
        config=config,
        worker_id=random.randint(0, 100),
        timeout_wait=300,
        reduced_actions=True
    )
    #Uncomment below if using frameskip
    #env = FrameSkip(env=env, skip=2)
    
    env = ObstacleTowerEvaluation(env, eval_seeds)
    
    return env

if __name__ == "__main__":
    # In this example we use the seeds used for evaluating submissions
    # to the Obstacle Tower Challenge.
    eval_seeds = [1001, 1002, 1003, 1004, 1005]


    # Create the ObstacleTowerEnv gym and launch ObstacleTower
    env = make_vec_env(lambda: create_env(), n_envs=1)

    # Uncomment if using FrameStack
    #env = VecFrameStack(env, n_stack=4)
    
    # Wrap the environment with the ObstacleTowerEvaluation wrapper
    # and provide evaluation seeds.

    # Replace with path to model zip
    model_path = "ppo_8_reduced_hyper.zip"


    model = PPO.load(model_path, env=env, device="cuda")
    #model = PPO("CnnPolicy", env, verbose=1)
    
    # We can run episodes (in this case with a random policy) until
    # the "evaluation_complete" flag is True.  Attempting to step or reset after
    # all of the evaluation seeds have completed will result in an exception.

    while not env.envs[0].evaluation_complete:
        episode_rew = run_episode(env, model)
        if episode_rew is None:
            break

    # Finally the evaluation results can be fetched as a dictionary from the
    # environment wrapper.
    print(env.envs[0].results)

    env.close()
