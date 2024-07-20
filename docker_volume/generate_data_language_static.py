"""Data collection script."""

import os
import hydra
import numpy as np
import random

import tasks
from dataset.dataset import store_to_dataset_language, load_dataset_language
from environments.environment import Environment
from tasks.utils import get_matrix


@hydra.main(config_path='/home/robot/docker_volume/configs', config_name='data')
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
    # record = cfg['record']['save_video']
    save_data = cfg['save_data']
    n_perspectives = cfg['n_perspectives']

    # Initialize scripted oracle agent and dataset.
    agent = task.create_oracle_agent(env)
    # Initialize dataset
    data_path = os.path.join(cfg['data_dir'], "language/simple", task.mode)
    synchronized_dataset = load_dataset_language(n_perspectives, data_path)
    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = synchronized_dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'valid':
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, valid, test")

    # Collect training data from oracle demonstrations
    while len(synchronized_dataset) < cfg['n']:
        seed += 2
        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'valid' and seed > (-1 + 10000):
            raise Exception(
                "!!! Seeds for valid set will overlap with the test set !!!")

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        env.set_task(task)
        # Reset environment
        # NOTE: diconnects bullet client to clear memory leakage from "p.loadTexture" inside task.reset()
        env.reset()

        done = False
        n_try = 0
        while not done and (n_try < task.n_tries):
            print(
                'Oracle demo: {}/{} | Seed: {} | Try: {}/{}'.format(len(synchronized_dataset) + 1, cfg['n'], seed, n_try + 1, task.n_tries))
            # Get initial observation right after restoration of environment
            obs, info = env.restore()
            if obs is not None:
                valid_obs = obs
            if info is not None:
                valid_info = info

            # Start video recording (NOTE: super slow)
            # if record:
            #     env.start_rec(f'{len(synchronized_dataset)+1:06d}')

            act = agent.act(obs, info)
            lang_goal = task.get_lang_goal()
            reward, done = env.step(act)
            steps = env.primitive_steps
            print(
                f'Total Reward: {reward:.3f} | Done: {done} | Goal: {lang_goal}')
            n_try += 1

        # End video recording
        # if record:
        #     env.end_rec()

        # Only save completed demonstrations.
        if save_data and reward > 0.99:
            observations = []
            for ob_c, a_cam in zip(valid_obs['color'], env.agent_cams):
                observation = {'color': ob_c,
                               'pose': get_matrix(a_cam['position'], a_cam['rotation']),
                               'intrinsics': a_cam['intrinsics']}
                observations.append(observation)

            grasp_pose = get_matrix(act[0][0], act[0][1])
            steps = [get_matrix(step[0], step[1]) for step in steps]
            # The last step (steps[-1]) should be equal to the grasp_pose
            # NOTE: info contains object id : (position, rotation, dimensions, urdf path)
            # store_to_dataset_language(synchronized_dataset,
            #                           observations, steps, grasp_pose, info=valid_info, language=lang_goal, seed=seed)


if __name__ == '__main__':
    main()
