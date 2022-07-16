import gym
import sys, os

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper


from file_helper import correct_folder_name

#define the environment to use
env_id = 'ALE/Tetris-v5'

'''
Tetris env is 
Actions Discrete(5)
Observations Box(210, 160, 3), uint8)

Supported Algorithms SB1: A2C, ACER, ACKTR, GAIL, DQN, PPO, TRPO
Supported Algorithms SB3: A2C, DQN, PPO

'''

#define log folder
env_folder_name = correct_folder_name(env_id)
log_dir = f"./log/{env_folder_name}"

#define checkpoint folder
checkpoint_dir = f"./checkpoints/{env_folder_name}"

training_steps = 1_000_000


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


def get_env():
    # Create environment
    env = make_atari_env(env_id, n_envs=4, seed=0)
    #env = AtariWrapper(env, noop_max=10,clip_reward=False)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)
    return env

def train():
    
    env = get_env()

    # Instantiate the agent
    model = DQN('MlpPolicy', env, tensorboard_log=log_dir, verbose=1)
    #model = DQN('CnnPolicy', env, tensorboard_log=log_dir, verbose=1)
    #model = PPO('CnnPolicy', env, tensorboard_log=log_dir, verbose=1)

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)
    
    # Train the agent
    model.learn(total_timesteps=training_steps, callback=callback)

    # Save the agent
    model.save("dqn_{env_folder_name}")
    #model.save(f"ppo_{env_folder_name}")
    
    del model  # delete trained model to demonstrate loading

def train_her():
    from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
    from stable_baselines3.common.noise import NormalActionNoise

    env = get_env()

    # Create 4 artificial transitions per real transition
    n_sampled_goal = 4

    # SAC hyperparams:
    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",
            # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
            # we have to manually specify the max number of steps per episode
            max_episode_length=5000,
            online_sampling=True,
            ),
        verbose=1,
        # buffer_size=int(1e6),
        # learning_rate=1e-3,
        # gamma=0.95,
        # batch_size=256,
        # policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log=log_dir,
    )

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)

    model.learn(total_timesteps=training_steps, callback=callback)

    model.save(f"her_{env_folder_name}")
    
    del model  # delete trained model to demonstrate loading

def manual():

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
    # env = gym.make(env_id, render_mode="human")
    # render = True

    env = make_atari_env(env_id, n_envs=1, seed=0)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one

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
        
        #if render:
        env.render()
        
        if dones:
            obs = env.reset()
            print(f"Total reward: {total_reward}")
            total_reward = 0.0

def continue_training():

    env = make_atari_env(env_id, n_envs=4, seed=0)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one

    model = PPO.load(f"ppo_{env_folder_name}", env=env, tensorboard_log=log_dir)
    
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=checkpoint_dir)
    
    # Train the agent
    model.learn(total_timesteps=training_steps, callback=callback)

    # Save the agent
    #model.save("dqn_{env_folder_name}")
    model.save(f"ppo_{env_folder_name}")

def main():
    if "train" in sys.argv:
        train()
    elif "manual" in sys.argv:
        manual()
    elif "run" in sys.argv:
        run()
    elif "continue" in sys.argv:
        continue_training()
    else:
        print("Please specify either 'train' or 'run'.")

if __name__ == "__main__":
    main()