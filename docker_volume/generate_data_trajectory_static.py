"""Data collection script."""

import os
import hydra
import numpy as np
import random

from simulation.tasks import names as task_names
from dataset.dataset import store_to_dataset_trajectory, load_dataset_trajectory
from simulation.environments.environment import Environment
from simulation.tasks.utils import get_matrix


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
    save_data = cfg['save_data']
    n_perspectives = cfg['n_perspectives']

    # Initialize scripted oracle agent and dataset.
    agent = task.create_oracle_agent(env)

    data_path = os.path.join(cfg['data_dir'], "nerf/simple", task.mode)
    # dataset = Dataset(data_path, cfg["dataset"])
    synchronized_dataset = load_dataset_trajectory(n_perspectives, data_path)

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = synchronized_dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'valid':  # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, valid, test")

    # Collect training data from oracle demonstrations
    while len(synchronized_dataset) < cfg['n']:
        total_reward = 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print(
            'Oracle demo: {}/{} | Seed: {}'.format(len(synchronized_dataset) + 1, cfg['n'], seed))
        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'valid' and seed > (-1 + 10000):
            raise Exception(
                "!!! Seeds for valid set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{len(synchronized_dataset)+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            lang_goal = info['lang_goal']
            _obs, reward, done, info = env.step(act)
            steps = env.primitive_steps
            total_reward += reward
            print(
                f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            # dataset.add(seed, episode)
            observations = []
            for ob_c, a_cam in zip(obs['color'], env.agent_cams):
                observation = {'color': ob_c,
                               'pose': get_matrix(a_cam['position'], a_cam['rotation']),
                               'intrinsics': a_cam['intrinsics']}
                observations.append(observation)

            grasp_pose = get_matrix(act['pose0'][0], act['pose0'][1])
            steps = [get_matrix(step[0], step[1]) for step in steps]
            # The last step (steps[-1]) should be equal to the grasp_pose
            # NOTE: the argument info is not the same as the info returned by env.step()
            store_to_dataset_trajectory(synchronized_dataset,
                                        observations, steps, grasp_pose, info=None, seed=seed)


if __name__ == '__main__':
    main()
