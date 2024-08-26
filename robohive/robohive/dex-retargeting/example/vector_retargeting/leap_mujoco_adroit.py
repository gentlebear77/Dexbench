import multiprocessing
from pathlib import Path
from queue import Empty
from typing import Optional

import requests
import threading
import os, sys, inspect, threading, time, datetime

import cv2
import numpy as np

import sapien
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

target_pos = np.zeros((21,3))
wrist_transition = np.zeros(3)
wrist = np.zeros(6)

def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray):
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

def Vector2Numpy(vector):
    return np.array([vector.x, vector.y,vector.z])

def Quaternion2Numpy(vector):
    return np.array([vector.x, vector.y,vector.z,vector.w])

def quaternion_to_euler_array(quaternions):
    eulers = []

    w, x, y, z = quaternions
    
    # calucate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # calucate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  
    else:
        pitch = np.arcsin(sinp)
    
    # calucate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)/10

    return roll, pitch , yaw

def map(event):
    global target_pos
    global wrist_transition
    hand = event.hands[0]
    # actions: joints
    # WRJ1: wrist
    #print(frame.hands[0].wrist_position
    joint_pos = np.zeros((21,3))
    joint_pos[0] = Vector2Numpy(hand.arm.next_joint)
    print("joint_pos[0]: ", joint_pos[0])
    # joint_pos[0][1] /=5
    wrist_transition = joint_pos[0] / 150.0
    wrist_transition[2] = -wrist_transition[2]
     
    wrist_transition[1] = wrist_transition[1] - 0.9
    wrist_transition[0] = -wrist_transition[0]
    # WRJ0: palm
    # joint_pos[1] = Vector2Numpy(hand.palm.position)

    # self.thumb, self.index, self.middle, self.ring, self.pinky
    # bones: self.metacarpal, self.proximal, self.intermediate, self.distal
    i = 1
    for index_digit in range(0, 5):
        digit = hand.digits[index_digit]
        for index_bone in range(0, 4):
            bone = digit.bones[index_bone]
            joint_pos[i] = Vector2Numpy(bone.next_joint)
            i = i+1
    joint_pos = (joint_pos -joint_pos[0]) / 1000
#     leap2mediapipe = np.array(
#     [
#         [0, -1, 0],
#         [-1, 0, 0],
#         [0, 0, -1],
#     ]
# )
#     joint_pos = joint_pos @ leap2mediapipe
#     mediapipe_wrist_rot = estimate_frame_from_hand_points(joint_pos)
#     operator2mano = np.array(
#     [
#         [0, 0, -1],
#         [-1, 0, 0],
#         [0, 1, 0],
#     ]     
# )
#     target_pos = joint_pos @ mediapipe_wrist_rot @ operator2mano
    # # print("leap1: ", joint_pos[1])
    from scipy.spatial.transform import Rotation as R
    r1 = R.from_euler('z', -90, degrees=True).as_matrix()
    r2 = R.from_euler('x', 180, degrees=True).as_matrix()
    # # joint_pos[:, 2] *= -1
    target_pos= np.matmul(r2,np.matmul(joint_pos,r1).T).T
    # print("target_pos: ", target_pos)
    wrist_qrotation = Quaternion2Numpy(hand.arm.rotation)
    wrist_rotation = quaternion_to_euler_array(wrist_qrotation)
    # print("wrist_transition:     ", wrist_transition) 
    # print("wrist_rotation: ", wrist_rotation)
    # wrist position and rotation

    # wrist = np.concatenate((wrist_transition, wrist_rotation), axis=None)
    # print("wrist: ", wrist)
    # time.sleep(1 / 30.0)


class LeapListener(leap.Listener):
    def on_connection_event(self, event):
        print("Connected")		
    
    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        # print(f"Frame {event.tracking_frame_id} with {len(event.hands)} hands.")
        if len(event.hands) > 0:
            # self.canvas.render_hands(event)	
            map(event)
         
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
    listener = LeapListener()
    connection = leap.Connection()
    
    # start listening
    connection.add_listener(listener)

    running = True
    
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    print(config_path)

    hand_type = "Right" if "right" in config_path.lower() else "Left"

    envw = gym.make("adroitbox-v0")
    env = envw.unwrapped
    env.reset()

    # sapien.render.set_viewer_shader_dir("default")
    # sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # # Setup
    # scene = sapien.Scene()
    # render_mat = sapien.render.RenderMaterial()
    # render_mat.base_color = [0.06, 0.08, 0.12, 1]
    # render_mat.metallic = 0.0
    # render_mat.roughness = 0.9
    # render_mat.specular = 0.8
    # scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # # Lighting
    # scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    # scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    # scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    # scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    # scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # # Camera
    # cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    # cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    # viewer = Viewer()
    # viewer.set_scene(scene)
    # viewer.control_window.show_origin_frame = False
    # viewer.control_window.move_speed = 0.01
    # viewer.control_window.toggle_camera_lines(False)
    # viewer.set_camera_pose(cam.get_local_pose())

    # # Load robot and set it to a good pose to take picture
    # loader = scene.create_urdf_loader()
    # filepath = Path(config.urdf_path)
    # robot_name = filepath.stem
    # loader.load_multiple_collisions_from_file = True
    # if "ability" in robot_name:
    #     loader.scale = 1.5
    # elif "dclaw" in robot_name:
    #     loader.scale = 1.25
    # elif "allegro" in robot_name:
    #     loader.scale = 1.4
    # elif "shadow" in robot_name:
    #     loader.scale = 0.9
    # elif "bhand" in robot_name:
    #     loader.scale = 1.5
    # elif "leap" in robot_name:
    #     loader.scale = 1.4
    # elif "svh" in robot_name:
    #     loader.scale = 1.5

    # if "glb" not in robot_name:
    #     filepath = str(filepath).replace(".urdf", "_glb.urdf")
    # else:
    #     filepath = str(filepath)

    # robot = loader.load(filepath)

    # if "ability" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))
    # elif "shadow" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.2]))
    # elif "dclaw" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))
    # elif "allegro" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.05]))
    # elif "bhand" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.2]))
    # elif "leap" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))
    # elif "svh" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # # Different robot loader may have different orders for joints
    # sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    # sapien_joint_names: 
    retargeting_joint_names = retargeting.joint_names
    print(retargeting_joint_names)
    mujoco_joint_names = env.get_joint_name()
    print(mujoco_joint_names)
    # retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)
    retargeting_to_mujoco = np.array([retargeting_joint_names.index(name) for name in mujoco_joint_names]).astype(int)
    print(retargeting_to_mujoco)
    # retargeting_to_mujoco = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29]

    # print(f"last qpos: {retargeting.last_qpos}")
    # sys.exit()
    #['joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_12.0', 'joint_1.0', 'joint_5.0', 'joint_9.0', 'joint_13.0', 'joint_2.0', 'joint_6.0', 'joint_10.0', 'joint_14.0', 'joint_3.0', 'joint_7.0', 'joint_11.0', 'joint_15.0'] 
    #['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0'] 
    #[ 0  8 12  4  1  9 13  5  2 10 14  6  3 11 15  7]
    #[ 6  7  8 17 21 12 25  9 18 22 13 26 10 19 23 14 27 11 20 24 15 28 16 29]
    # [ 6  7  8  9 10 11 17 18 19 20 21 22 23 24 12 13 14 15 16 25 26 27 28 29]

    # [ 0  1  2  3  4  5 11 12 13 14 15 16 17 18  6  7  8  9 10 19 20 21 22 23]
    retargeting_to_mujoco = [ 0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29]
    # sys.exit()
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while running:
            if target_pos is None:
                logger.warning(f"{hand_type} hand is not detected.")
            else:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                # print("sapien: ", target_pos[1])
                # indices: shadow (2, 15) allegro (2, 10)
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = target_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = target_pos[task_indices, :] - target_pos[origin_indices, :]
                
                # print("ref_value:", ref_value)
                #ref_value: shadow (15, 3) allegro (10, 3)
                # print("ref_value:", ref_value)
                # print("target_pos: ", target_pos)
                qpos_hand = retargeting.retarget(ref_value)
                # qpos = np.concatenate((wrist, qpos_hand[retargeting_to_mujoco]), axis=None)
                # print("qpos:", qpos)
                # initial = np.zeros(30)
                qpos_hand[:3] = wrist_transition
                qpos_hand[4:5] = 0.0
                env.step(qpos_hand[retargeting_to_mujoco])
                print("qpos_hand: ", qpos_hand)
                # env.sim.data.qpos[:30] = qpos_hand[retargeting_to_mujoco]
                # env.sim.forward()
                #qpos (24,) allegro (16,)
                # robot.set_qpos(qpos[retargeting_to_sapien])
            for _ in range(2):
                env.mj_render()
            # for _ in range(2):
            #     viewer.render()
                

if __name__ == "__main__":
    tyro.cli(main)