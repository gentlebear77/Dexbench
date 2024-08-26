#!/usr/bin/env python3
DESC = """
Vizualize model in a viewer\n
    - render forward kinematics if `qpos` is provided\n
    - simulate dynamcis if `ctrl` is provided\n
Example:\n
    - python utils/examine_sim.py --sim_path envs/arms/franka/assets/franka_reach_v0.xml --qpos "0, 0, -1, -1, 0, 0, 0, 0, 0"\n
    - python utils/examine_sim.py --sim_path envs/arms/franka/assets/franka_reach_v0.xml --ctrl "0, 0, -1, -1, 0, 0, 0, 0, 0"\n
"""

from mujoco import MjModel, MjData, mj_step, mj_forward, viewer
import click
import numpy as np
from robohive.utils import gym
import numpy as np
import pickle
import cv2
from robohive.envs import env_base

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', required= True)
@click.option('-p', '--policy_path', type=str, help='absolute path of the policy file', default=None)

def main(env_name, policy_path):
    envw = gym.make(env_name)
    env = envw.unwrapped
    env.reset()
    # Load replay data from pickle file
    with open(policy_path, 'rb') as f:
        replay_data = pickle.load(f)

    for step_data in replay_data:
        print(f"Replaying Step: {step_data['step']}")

        # Set the environment state to the saved state
        env.set_env_state(step_data['state_dict'])

        
        for _ in range(2):
            env.mj_render()
    del env


if __name__ == "__main__":

    main()
