"""Data collection script."""

import os
import hydra
import numpy as np
import random

from simulation.tasks import names as task_names
from dataset.utils import load_dataset_language
from simulation.environments.environment import Environment
from simulation.tasks import utils
import matplotlib.pyplot as plt


@hydra.main(config_path='/home/robot/docker_volume/simulation/configs', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = task_names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    n_perspectives = cfg['n_perspectives']

    # Initialize dataset
    data_path = os.path.join(cfg['data_dir'], "language/simple", task.mode)
    synchronized_dataset = load_dataset_language(n_perspectives, data_path)

    # NOTE: The way in which the grasps are shown is not ordered, because of the way files are found by os.listdir()
    # NOTE: It will obviously break things if the task is changed in any way between generation and validation
    # But since it is for validation only, this should suffice
    for idx in range(len(synchronized_dataset)):
        seed = synchronized_dataset.datasets['color'].seeds[idx]
        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        env.set_task(task)
        # Reset environment
        env.reset()

        print('Seed: {}'.format(seed))
        while not env.is_static:
            env.step_simulation()

        # Validate info
        info = synchronized_dataset.datasets["info"].read_sample(idx)
        if info is None:
            print("Info None")
        else:
            load_env_from_info()
            # NOTE: I need a methode to create an env object from an info
            print("Info not None")

        # Validate goal
        lang_goal = synchronized_dataset.datasets["language"].read_sample(idx)
        print(f'Goal: {lang_goal}')

        # Validate one image
        color = synchronized_dataset.datasets["color"].read_sample(idx)
        # Validate camera config of this one image
        cam_config = synchronized_dataset.datasets["camera_config"].read_sample(idx)[0][
            "pose"]
        pos = cam_config[:3, 3]
        rot = utils.mat_to_quat(cam_config[:3, :3])
        pose = (pos, rot)
        env.add_object(urdf='util/coordinate_axes.urdf',
                       pose=pose, category='fixed')
        plt.imshow(color[0])
        plt.show()

        # Validate trajectory
        steps = synchronized_dataset.datasets["trajectory"].read_sample(idx)[
            "trajectory"]
        for step in steps:
            pos = step[:3, 3]
            rot = utils.mat_to_quat(step[:3, :3])
            pose = (pos, rot)
            env.add_object(urdf='util/coordinate_axes.urdf',
                           pose=pose, category='fixed')

        # Validate grasp_pose
        hom_mat = synchronized_dataset.datasets["grasp_pose"].read_sample(idx)[
            "grasp_pose"]
        pos = hom_mat[:3, 3]
        rot = utils.mat_to_quat(hom_mat[:3, :3])
        act = [(pos, rot)]
        reward, done = env.step(act)

        # End video recording
        if record:
            env.end_rec()


def load_env_from_info():
    pass


if __name__ == '__main__':
    main()
