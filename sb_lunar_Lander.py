import gym
import sys, os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

env_id = "LunarLander-v2"
local_dir = f"sb_runs_{env_id}"

#define checkpoint folder
checkpoint_dir =  os.path.join(local_dir, f"./checkpoints/{env_id}")
log_dir = os.path.join(local_dir, f"./log/{env_id}")

# Create environment
env = gym.make(env_id)

mode = "PPO"
timesteps = 2_000_000


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'checkpoint_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

def train_dqn():
    # Instantiate the agent
    model = DQN('MlpPolicy', env, tensorboard_log=log_dir, verbose=1)

    # Train the agent
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)
    model.learn(total_timesteps=timesteps, callback=callback)
    # Save the agent
    folder = os.path.join(local_dir, "dqn_lunar")
    model.save(folder)
    del model  # delete trained model to demonstrate loading

def train_ppo():
    # Instantiate the agent
    model = PPO('MlpPolicy', env, tensorboard_log=log_dir, verbose=1)
    
    # Train the agent
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)
    model.learn(total_timesteps=timesteps, callback=callback)
    # Save the agent
    folder = os.path.join(local_dir, "ppo_lunar")
    model.save(folder)
    del model  # delete trained model to demonstrate loading

def train():

    if mode == "DQN":
        train_dqn()
    elif mode == "PPO":
        train_ppo()



def enjoy():
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    if mode == "DQN":
        folder = os.path.join(local_dir, "dqn_lunar")
        model = DQN.load(folder, env=env)
    elif mode == "PPO":
        folder = os.path.join(local_dir, "ppo_lunar")
        model = PPO.load(folder, env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

def main():
    if "train" in sys.argv:
        train()
    elif "run" in sys.argv:
        enjoy()
    else:
        print("Please specify either 'train' or 'run' or 'tensorboard'.")

if __name__ == "__main__":
    main()