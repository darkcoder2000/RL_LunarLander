import gym
import sys, os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from file_helper import correct_folder_name

env_list = ['LunarLander-v2']
env_list_with_render_mode = ['ALE/Tetris-v5']

#define the environment to use
env_id = 'ALE/Tetris-v5'

#define log folder
env_folder_name = correct_folder_name(env_id)
log_dir = f"./log/{env_folder_name}"

#define checkpoint folder
checkpoint_dir = f"./checkpoints/{env_folder_name}"


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


def train():
    # Create environment
    env = gym.make(env_id)

    # Instantiate the agent
    #model = DQN('MlpPolicy', env, tensorboard_log=log_dir, verbose=1)
    model = PPO('MlpPolicy', env, tensorboard_log=log_dir, verbose=1)

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)
    
    # Train the agent
    model.learn(total_timesteps=int(100000), callback=callback)

    # Save the agent
    #model.save("dqn_{env_folder_name}")
    model.save(f"ppo_{env_folder_name}")
    
    del model  # delete trained model to demonstrate loading

def run_maual_tetris():

    #https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/docs/environment.md
    '''
    0   Do nothing
    1	UP	Apply a Δ-movement upwards on the joystick
    2	RIGHT	Apply a Δ-movement rightward on the joystick
    3	LEFT	Apply a Δ-movement leftward on the joystick
    4	DOWN	Apply a Δ-movement downward on the joystick
    '''
    env = gym.make(env_id, render_mode="human")

    obs = env.reset()
    while True:
        #get cmd input
        cmd = input("Enter command: ")
        print(f"Command: {cmd}")

        if cmd == "6":
            cmd = 2
        elif cmd == "4":
            cmd = 3
        elif cmd == "2":
            cmd = 4
        elif cmd == "5":
            cmd = 1
        else:
            cmd = 0
        obs, rewards, dones, info = env.step(cmd)
        print(f"Reward: {rewards}")
        if dones:
            obs = env.reset()

def run():

    # Create environment
    if env_id in env_list:
        env = gym.make(env_id)
        render = False
    elif env_id in env_list_with_render_mode:
        env = gym.make(env_id, render_mode="human")
        render = True

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    #model = DQN.load("dqn_lunar", env=env)
    model = PPO.load(f"ppo_{env_folder_name}", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    total_reward = 0.0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print(f"Reward: {rewards}")
        total_reward += rewards
        
        if render:
            env.render()
        
        if dones:
            obs = env.reset()
            print(f"Total reward: {total_reward}")
            total_reward = 0.0


def main():
    if "train" in sys.argv:
        train()
    elif "run" in sys.argv:
        run_maual_tetris()
    else:
        print("Please specify either 'train' or 'run'.")

if __name__ == "__main__":
    main()