import collections
from robohive.utils import gym
import numpy as np

from robohive.envs import env_base
# NOTES:
#     1. why is qpos[0] not a part of the obs? ==> Hand translation isn't consistent due to randomization. Palm pos is a good substitute

# OBS_KEYS = ['hand_jnt', 'latch_pos', 'box_pos', 'palm_pos', 'handle_pos', 'reach_err', 'box_open'] # DAPG
DEFAULT_OBS_KEYS = ['hand_jnt', 'palm_pos', 'box_top']
# RWD_KEYS = ['reach', 'open', 'smooth', 'bonus'] # DAPG
DEFAULT_RWD_KEYS_AND_WEIGHTS = {'reach':1.0, 'open':1.0, 'bonus':1.0}

class AllegroBoxV0(env_base.MujocoEnv):

    DEFAULT_CREDIT = """\
    Dexbench environment for Allegro hand
    """

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(env_credits=self.DEFAULT_CREDIT, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)


    def _setup(self,
            frame_skip=5,
            reward_mode="dense",
            obs_keys=DEFAULT_OBS_KEYS,
            weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs
            ):
        

        # ids
        sim = self.sim
        self.grasp_sid = sim.model.body_name2id('palm')
        self.box_bid = sim.model.body_name2id('boxbodytop')
        # change actuator sensitivity
        # sim.model.actuator_gainprm[sim.model.actuator_name2id('A_WRJ1'):sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        # sim.model.actuator_gainprm[sim.model.actuator_name2id('A_FFJ3'):sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        # sim.model.actuator_biasprm[sim.model.actuator_name2id('A_WRJ1'):sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        # sim.model.actuator_biasprm[sim.model.actuator_name2id('A_FFJ3'):sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])
        # scales
        self.act_mid = np.mean(sim.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(sim.model.actuator_ctrlrange[:,1]-sim.model.actuator_ctrlrange[:,0])

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        # self.init_qpos = np.zeros(self.init_qpos.shape)
        self.init_qpos = self.sim.data.qpos
        self.init_qvel = np.zeros(self.init_qvel.shape)


    def get_obs_dict(self, sim):
        # qpos for hand, xpos for obj, xpos for target
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[:-6].copy()
        obs_dict['box_top'] = sim.data.body_xpos[self.box_bid].copy()
        obs_dict['palm_pos'] = sim.data.body_xpos[self.grasp_sid].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        # reach_dist = np.linalg.norm(self.obs_dict['reach_err'], axis=-1)
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -0.1),
            ('open',    -0.1),
            ('bonus',   2),
            # Must keys
            ('sparse',  True),
            ('solved',  False),
            ('done',    False),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.sim.reset()
        qp = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qv = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(reset_pos=qp, reset_vel=qv, **kwargs)
        # print(f"qp_size: {qp.size}")
        # print(f"Box pos_origin: {self.sim.model.body_pos[self.box_bid]}")
        self.sim.model.body_pos[self.box_bid,0] = self.np_random.uniform(low=-0.1, high=0.2)
        self.sim.model.body_pos[self.box_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        # print(f"Box pos: {self.sim.model.body_pos[self.box_bid]}")
        self.sim.forward()
        return self.get_obs()


    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        box_body_pos = self.sim.model.body_pos[self.box_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, box_body_pos= box_body_pos)


    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.sim.set_state(qpos=qp, qvel=qv)
        self.sim.model.body_pos[self.box_bid] = state_dict['box_body_pos']
        self.sim.forward()

    def get_joint_name(self):
        """
        Get joint name from joint id
        """
        leap_joint=[]
        joint_count = self.sim.model.njnt
        for i in range (joint_count):
            joint_name = self.sim.model.id2name(i, 'joint')
            # print(f"joint_nam: {joint_name}")
            # all the num of joint need to minus 1 to meet leap motion joint num
            import re
            new_name = re.sub(r'\d+', lambda x: str(int(x.group()) + 1), joint_name)
            leap_joint.append(new_name)
        # print(self.sim.model.jnt_range)
        return leap_joint[6:30]
        print(leap_joint[7:])
    
    def check_success(self):
        """
        Check if the task is successful
        """
        # if self.obs_dict['reach_err'] < 0.1:
        #     return True
        # else:
        #     return False
        return False