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

@click.command(help=DESC)
@click.option('-s', '--sim_path', type=str, help='environment to load', required= True)
@click.option('-q', '--qpos', type=str, help='joint position', default=None)
@click.option('-c', '--ctrl', type=str, help='actuator position', default=None)
@click.option('-h', '--horizon', type=int, help='time (s) to simulate', default=100)

def main(sim_path, qpos, ctrl, horizon):
    model = MjModel.from_xml_path(sim_path)
    data = MjData(model)
    
    viewer.launch(model, data)
    while data.time<horizon:
        # print("in loop")
        if qpos is not None:
            print("inside")
            data.qpos[:] = np.array(qpos.split(','), dtype=np.float)
            mj_forward(model, data)
            data.time += model.opt.timestep

        elif ctrl is not None:
            data.ctrl[:] = np.array(ctrl.split(','), dtype=np.float)
            mj_step(model, data)

    
# "1, 0.3, 0, 0.70710678, 0, 0, 0.70710678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"
# "-0.25, 0, -0.3, -0.75, -0.75, -0.75, -0.524, -0.79, -0.44, 0, 0, 0, -0.44, 0, 0, 0, -0.44, 0, 0, 0, 0, -0.44, 0, 0, 0, -1, 0, -0.26, -0.52, -1.57"

if __name__ == '__main__':
    main()
