"""Data collection script."""

import os
import hydra
import numpy as np
import random

import tasks
from datasets.dataset import store_to_dataset_nerf, load_dataset_nerf
from environments.environment import Environment
from tasks.utils import get_matrix


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
    save_data = cfg['save_data']
    n_perspectives = cfg['n_perspectives']

    # Initialize scripted oracle agent and dataset.
    agent = task.create_oracle_agent(env)

    data_path = os.path.join(cfg['data_dir'], "nerf/simple", task.mode)
    # dataset = Dataset(data_path, cfg["dataset"])
    synchronized_dataset = load_dataset_nerf(n_perspectives, data_path)

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
    # while synchronized_dataset.n_episodes < cfg['n']:
    while len(synchronized_dataset) < cfg['n']:
        # episode = []
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
            # env.start_rec(f'{synchronized_dataset.n_episodes+1:06d}')
            env.start_rec(f'{len(synchronized_dataset)+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            # episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            _obs, reward, done, info = env.step(act)
            total_reward += reward
            print(
                f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        # episode.append((obs, None, reward, info))

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
            # NOTE: the argument info is not the same as the info returned by env.step()
            store_to_dataset_nerf(synchronized_dataset,
                                  observations, info=None, seed=seed)


if __name__ == '__main__':
    main()
