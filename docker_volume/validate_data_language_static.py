"""Data collection script."""

import os
import hydra
import numpy as np
import random

import tasks
from datasets.dataset import load_dataset_language
from environments.environment import Environment
from tasks import utils
import matplotlib.pyplot as plt


@hydra.main(config_path='/home/robot/shared_docker_volume/configs', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    n_perspectives = cfg['n_perspectives']

    # Initialize dataset
    data_path = os.path.join(cfg['data_dir'], "language/simple", task.mode)
    synchronized_dataset = load_dataset_language(n_perspectives, data_path)

    # NOTE: The way in which the grasps are shown is not ordered, because of the way files are found by os.listdir()
    # NOTE: It will obviously break things if the task is changed in any way between generation and validation
    # But since it is for validation only, this should suffice
    for seed in synchronized_dataset.seeds:
        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'valid' and seed > (-1 + 10000):
            raise Exception(
                "!!! Seeds for valid set will overlap with the test set !!!")
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
        info = synchronized_dataset.datasets["info"].read_sample(seed)
        if info is None:
            print("Info None")
        else:
            load_env_from_info()
            # NOTE: I need a methode to create an env object from an info
            print("Info not None")

        # Validate goal
        lang_goal = synchronized_dataset.datasets["language"].read_sample(seed)
        print(f'Goal: {lang_goal}')

        # Validate one image
        color = synchronized_dataset.datasets["color"].read_sample(seed)
        # Validate camera config of this one image
        cam_config = synchronized_dataset.datasets["camera_config"].read_sample(seed)[0][
            "pose"]
        pos = cam_config[:3, 3]
        rot = utils.mat_to_quat(cam_config[:3, :3])
        pose = (pos, rot)
        env.add_object(urdf='util/coordinate_axes.urdf',
                       pose=pose, category='fixed')
        plt.imshow(color[0])
        plt.show()

        # Validate trajectory
        steps = synchronized_dataset.datasets["trajectory"].read_sample(seed)[
            "trajectory"]
        for i in range(0, len(steps), 10):
            pos = steps[i][:3, 3]
            rot = utils.mat_to_quat(steps[i][:3, :3])
            pose = (pos, rot)
            env.add_object(urdf='util/coordinate_axes.urdf',
                           pose=pose, category='fixed')

        # Validate grasp_pose
        hom_mat = synchronized_dataset.datasets["grasp_pose"].read_sample(seed)[
            "grasp_pose"]
        pos = hom_mat[:3, 3]
        rot = utils.mat_to_quat(hom_mat[:3, :3])
        _act = (pos, rot)
        # Bring into proper format for pick and place primitive
        act = {"pose0": _act, "pose1": _act}
        _obs, reward, done, _info = env.step(act)

        # End video recording
        if record:
            env.end_rec()


def load_env_from_info(info):
    pass


if __name__ == '__main__':
    main()
