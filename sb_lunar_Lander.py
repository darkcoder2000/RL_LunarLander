import gym
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env_id = "LunarLander-v2"
local_dir = f"sb_runs_{env_id}"

# Create environment
env = gym.make(env_id)

def train():
    # Instantiate the agent
    model = DQN('MlpPolicy', env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(200000))
    # Save the agent
    model.save("dqn_lunar")
    del model  # delete trained model to demonstrate loading


def enjoy():
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load("dqn_lunar", env=env)

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