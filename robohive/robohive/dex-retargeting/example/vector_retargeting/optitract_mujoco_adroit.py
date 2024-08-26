import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Optional

import requests
import threading
import os, sys, inspect, threading, time, datetime

import cv2
import numpy as np

import pickle
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
# from single_hand_detector import SingleHandDetector

from robohive.utils import gym

from robohive.envs.dexbench.adroit_box_v0 import AdroitBoxV0
import leap
from IPython import embed
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray 

from robohive.utils import quat_math


target_pos = np.zeros(20)
wrist_transition = np.zeros(3)
wrist_rotation = np.zeros(3)

def Vector2Numpy(vector):
    return np.array([vector.x, vector.y,vector.z])

def Quaternion2Numpy(vector):
    return np.array([vector.x, vector.y,vector.z,vector.w])

wrist_transition_initial = None
wrist_rotation_initial = None


def degrees_to_radians(degrees):
    degree_array = np.array(degrees)
    radians = degree_array * (np.pi / 180)
    return radians

def pose_callback(msg):
    # Print the received PoseStamped data
    # rospy.loginfo(f"Received PoseStamped data:\n"
    #               f"Data: {msg.data}\n")
    global target_pos
    target_pos = np.array(degrees_to_radians(msg.data))
    # print("target_pos: ", target_pos)   

def smooth_transition(current_angle, previous_angle, alpha=0.1):
    # Simple linear interpolation between previous and current angles
    return previous_angle * (1 - alpha) + current_angle * alpha

def wrist_callback(msg):
    # Print the received PoseStamped data
    # rospy.loginfo(f"Received PoseStamped data:\n"
    #               f"Position - x: {msg.pose.position.x}, y: {msg.pose.position.y}, z: {msg.pose.position.z}\n"
    #               f"Orientation - x: {msg.pose.orientation.x}, y: {msg.pose.orientation.y}, "
    #               f"z: {msg.pose.orientation.z}, w: {msg.pose.orientation.w}")
    global wrist_transition, wrist_rotation, wrist_transition_initial, wrist_rotation_initial
    if wrist_transition_initial is None:
        wrist_transition_initial = np.array([msg.pose.position.x, msg.pose.position.y, -msg.pose.position.z])
    # print("wrist_initial: ", wrist_initial)
    previous_wrist_rotation = np.array([0, 0, 0])

    if wrist_rotation_initial is None:
        wrist_rotation_initial = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        previous_wrist_rotation = quat_math.quat2euler(wrist_rotation_initial)
    wrist_transition = np.array([msg.pose.position.x, msg.pose.position.y, -msg.pose.position.z]) - wrist_transition_initial
    # print("wrist_transition: ", wrist_transition)
    wrist_rotation_1 = quat_math.quat2euler(quat_math.diffQuat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], wrist_rotation_initial))
    wrist_rotation = smooth_transition(wrist_rotation_1, previous_wrist_rotation)
    previous_wrist_rotation = wrist_rotation
    # wrist_rotation = quat_math.quat2euler([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    # print("wrist_rotation: ", wrist_rotation)
    
def qpos_control_map(qpos, upper, lower):
    return 2 * (qpos - upper) / (lower - upper) - 1

class DataCollector:
    def __init__(self, env, save_dir='/home/gentlebear/dexbench/robohive/robohive/rawdata', image_save_format='png'):
        self.env = env
        self.save_dir = save_dir
        self.image_save_format = image_save_format
        os.makedirs(save_dir, exist_ok=True)
        self.data = []

    def collect_data(self, step):
        obs_dict = self.env.get_obs_dict(self.env.sim)
        state_dict = self.env.get_env_state()
        
        # Save observation and state data
        data_entry = {
            'step': step,
            'obs_dict': obs_dict,
            'state_dict': state_dict,
            # 'image_path': image_path,
        }
        self.data.append(data_entry)

    def save_data(self):
        # Save the collected data as a pickle file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        data_path = os.path.join(self.save_dir, f'collected_data_{timestamp}.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self.data, f)

def main(
    robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType, camera_path: Optional[str] = None
    ):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = str(get_default_config_path(robot_name, retargeting_type, hand_type))
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
        # create the post listener and controller
    # listener = LeapListener()
    # connection = leap.Connection()
    
    # # start listening
    # connection.add_listener(listener)

    running = True
    
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"

    envw = gym.make("allegrobox-v0")
    env = envw.unwrapped
    env.reset()

    # sapien.render.set_viewer_shader_dir("default")
    # sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    retargeting_joint_names = retargeting.joint_names
    # print(retargeting_joint_names)
    mujoco_joint_names = env.get_joint_name()
    # print(mujoco_joint_names)
    # retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)
    # retargeting_to_mujoco = np.array([retargeting_joint_names.index(name) for name in mujoco_joint_names]).astype(int)
    # print(retargeting_to_mujoco)
    # retargeting_to_mujoco = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29]

    # print(f"last qpos: {retargeting.last_qpos}")
    # sys.exit()
    #['joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_12.0', 'joint_1.0', 'joint_5.0', 'joint_9.0', 'joint_13.0', 'joint_2.0', 'joint_6.0', 'joint_10.0', 'joint_14.0', 'joint_3.0', 'joint_7.0', 'joint_11.0', 'joint_15.0'] 
    #['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0'] 
    #[ 0  8 12  4  1  9 13  5  2 10 14  6  3 11 15  7]
    #[ 6  7  8 17 21 12 25  9 18 22 13 26 10 19 23 14 27 11 20 24 15 28 16 29]
    # [ 6  7  8  9 10 11 17 18 19 20 21 22 23 24 12 13 14 15 16 25 26 27 28 29]
    collecter = DataCollector(env)

    # retargeting_to_mujoco = [ 0, 1, 2, 5, 3, 4, 11, 12, 13, 10, 15, 16, 17, 14, 19, 20, 21, 18, 6, 7, 9, 8]
    retargeting_to_mujoco = [ 0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29]
    # sys.exit()
    rospy.init_node('pose_listener', anonymous=True)

    # Subscribe to the /natnet_ros/Test/pose topic
    rospy.Subscriber("/natnet_ros/right_hand/pose", PoseStamped, wrist_callback)
    
    rospy.Subscriber("/manus_glove_data_right", Float32MultiArray, pose_callback)
    duration = 10
    start_time = time.time()
    total_sub_steps = 0
    while not rospy.is_shutdown() and time.time() - start_time < duration:
        retargeting_type = retargeting.optimizer.retargeting_type
        joint_limits = retargeting.joint_limits
        indices = retargeting.optimizer.target_link_human_indices
        # print("sapien: ", target_pos[1])
        # indices: shadow (2, 15) allegro (2, 10)

        # if retargeting_type == "POSITION":
        #     indices = indices
        #     ref_value = target_pos[indices, :]
        # else:
        #     origin_indices = indices[0, :]
        #     task_indices = indices[1, :]
        #     ref_value = target_pos[task_indices, :] - target_pos[origin_indices, :]
        
        # print("ref_value:", ref_value)
        #ref_value: shadow (15, 3) allegro (10, 3)
        # print("ref_value:", ref_value)
        # print("target_pos: ", target_pos)
        # qpos_hand = retargeting.retarget(ref_value)
        # qpos = np.concatenate((wrist, qpos_hand[retargeting_to_mujoco]), axis=None)
        # print("qpos:", qpos)

        # initial = np.zeros(30)

        ref_value = np.zeros(30)
        ref_value[:3] = wrist_transition
        ref_value[3:6] = wrist_rotation
        ref_value[8:] = target_pos
        qpos = qpos_control_map(ref_value, joint_limits[:, 0], joint_limits[:, 1])
        qpos_hand = np.clip(qpos, -1, 1)
        # print("ref_value:", ref_value)
        env.step(qpos_hand[retargeting_to_mujoco])
        collecter.collect_data(total_sub_steps)
        total_sub_steps += 1
        for _ in range(2):
            env.mj_render()
        
        # for _ in range(2):
        #     viewer.render()
    collecter.save_data()       
    del env
    
                

if __name__ == "__main__":
    tyro.cli(main)