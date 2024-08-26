from robohive.utils import gym; register=gym.register

from robohive.envs.env_variants import register_env_variant
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Adroit Envs")

# ==================================================================================
# V1 envs:
#   - env_base class independent of mjrl, making it self contained
#   - Updated obs such that rwd are recoverable from obs
#   - Vectorized obs and rwd calculations
# ==================================================================================

# Swing the door open
register(
    id='adroitbox-v0',
    entry_point='robohive.envs.dexbench:AdroitBoxV0',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/adroit_tasks/Adroit_box.xml',
    }
)
from robohive.envs.dexbench.adroit_box_v0 import AdroitBoxV0

register(
    id='adroitmulti-v0',
    entry_point='robohive.envs.dexbench:AdroitMultiV0',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/adroit_tasks/Adroit_multi.xml',
    }
)
from robohive.envs.dexbench.adroit_multi_v0 import AdroitMultiV0

register(
    id='allegrobox-v0',
    entry_point='robohive.envs.dexbench:AllegroBoxV0',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/allegro_tasks/Allegro_box.xml',
    }
)
from robohive.envs.dexbench.allegro_box_v0 import AllegroBoxV0

register(
    id='allegromulti-v0',
    entry_point='robohive.envs.dexbench:AllegroMultiV0',
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/assets/allegro_tasks/Allegro_multi.xml',
    }
)
from robohive.envs.dexbench.allegro_multi_v0 import AllegroMultiV0

# Reach to random target using visual inputs
def register_visual_envs(env_name, encoder_type):
    register_env_variant(
        env_id='{}-v0'.format(env_name),
        variant_id='{}_v{}-v0'.format(env_name, encoder_type),
        variants={
                # add visual keys to the env
                'visual_keys':[
                    "rgb:vil_camera:224x224:{}".format(encoder_type),
                    "rgb:view_1:224x224:{}".format(encoder_type),
                    "rgb:view_4:224x224:{}".format(encoder_type)],
                # override the obs to avoid accidental leakage of oracle state info while using the visual envs
                # using time as dummy obs. time keys are added twice to avoid unintended singleton expansion errors.
                "obs_keys": ['time', 'time'],
                # add proprioceptive data - proprio_keys to configure, env.get_proprioception() to access
                'proprio_keys':
                    ['hand_jnt'],
        },
        silent=True
    )

for env_name in ["adroitbox","adroitmulti","allegrobox"]:
    for enc in ["r3m18", "r3m34", "r3m50", "rrl18", "rrl34", "rrl50", "2d", "vc1s"]:
        register_visual_envs(env_name, enc)
