import gymnasium as gym
import highway_env
import torch
import numpy as np
from rl_agents.agents.common.factory import load_agent
from rl_agents.trainer.evaluation import Evaluation

from algorithms.agents.DQN.model import model

env_config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": [
            "presence",
            "x",
            "y",
            "vx",
            "vy",
            "cos_h",
            "sin_h"
        ],
        "absolute": False
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

agent_config= {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron",
        "layers": [256, 256]
    },
    "double": False,
    "loss_function": "l2",
    "optimizer": {
        "lr": 5e-4
    },
    "gamma": 0.8,
    "n_steps": 1,
    "batch_size": 32,
    "memory_capacity": 15000,
    "target_update": 50,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 6000,
        "temperature": 1.0,
        "final_temperature": 0.05
    }
}

def envrionment():
    env = gym.make("highway-v0",render_mode='human')
    env.unwrapped.configure(env_config)
    return env

def import_model(env):
    # agent = load_agent(agent_config, env)
    state, info = env.reset()
    agent = model(env.action_space,state)
    return agent

def run(train = False):
    env = envrionment()
    agent = import_model(env)

    for _ in range(100):
        done = terminal = False
        state, info = env.reset()

        while not (terminal or done):
            action = env.env.unwrapped.action_type.actions_indexes["IDLE"]
            # action = agent.step(state)
            next_state, reward, terminal, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, terminal, info)
            state = next_state
            env.render()


if __name__ == "__main__":
    run()