# Copyright 2019 Zihan Ding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------------------------------------------
# Modifications Copyright 2025 Changyu Hu
#
# This file has been significantly modified from its original version in
# the Popular-RL-Algorithms library. The original license and copyright
# notices are retained above.
#
# The modifications are provided under the terms of the license of this project.
# --------------------------------------------------------------------------------
 
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from env import LBSEnv, KMeansEnv
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import torch.multiprocessing as mp
from sac import SAC_Trainer, sac_worker, ShareParameters
from sac import ReplayBuffer, RewardBuffer
from utils import create_folder

import torch

torch.multiprocessing.set_start_method('forkserver', force=True)


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train',
                        action='store_true', default=False)
    parser.add_argument('--test', dest='test',
                        action='store_true', default=False)
    parser.add_argument('--gpus', dest='gpus', default='0')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        action='store_true', default=False)
    parser.add_argument('--cfg_path', type=str, default='./config.json')

    return parser.parse_args()


def plot(rewards, path):
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig(path)
    # plt.show()
    plt.close()


def main():
    args = arg_parser()
    cfg_path = args.cfg_path
    cfg = json.load(open(cfg_path, 'r'))
    model_name = cfg["model_name"]
    experiment_name = cfg["experiment_name"]
    out_dir = Path(__file__).parent.parent / "output" / \
        experiment_name / model_name
    create_folder(out_dir, exist_ok=True)
    out_dir = str(out_dir)

    replay_buffer_size = 1e6
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    BaseManager.register('RewardBuffer', RewardBuffer)
    manager = BaseManager()
    manager.start()
    # share the replay buffer through manager
    replay_buffer = manager.ReplayBuffer(replay_buffer_size)

    reward_buffer = manager.RewardBuffer()

    max_episodes = cfg['max_episodes']
    total_time = cfg['total_time']
    interval = cfg['interval']
    max_steps = int(total_time / interval)
    batch_size = cfg['batch_size']
    # update_itr = 256
    update_itr = cfg['update_itr']
    AUTO_ENTROPY = cfg['AUTO_ENTROPY']
    DETERMINISTIC = cfg['DETERMINISTIC']
    hidden_dim = cfg['hidden_dim']
    explore_steps = cfg['explore_steps']
    rewards = []
    action_dim = cfg['action_size']
    dim = cfg['dim']
    env_type = cfg['env_type']
    if env_type == 'LBS':
        Env = LBSEnv
    elif env_type == 'KMeans':
        Env = KMeansEnv
    else:
        raise ValueError('Invalid environment type')
    state_dim = Env.get_state_dim(dim, action_dim)

    sac_trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_dim)
    if args.train:
        rewards = []
        if args.checkpoint:
            sac_trainer.load_model(out_dir + "/lbmsac_best")
            rewards = np.load(out_dir + "/rewards.npy").tolist()
            print("previous episodes: ", len(rewards))

        gpu_list = args.gpus.split(",")
        num_workers = len(gpu_list)
        for worker_id in range(num_workers):
            assert int(gpu_list[worker_id]
                       ) >= 0, "GPU id should be non-negative"
            assert int(gpu_list[worker_id]) < torch.cuda.device_count(
            ), "GPU id should be less than the number of GPUs"

        # share the global parameters in multiprocessing
        sac_trainer.soft_q_net1.share_memory()
        sac_trainer.soft_q_net2.share_memory()
        sac_trainer.target_soft_q_net1.share_memory()
        sac_trainer.target_soft_q_net2.share_memory()
        sac_trainer.policy_net.share_memory()
        ShareParameters(sac_trainer.soft_q_optimizer1)
        ShareParameters(sac_trainer.soft_q_optimizer2)
        ShareParameters(sac_trainer.policy_optimizer)
        ShareParameters(sac_trainer.alpha_optimizer)

        # used for get rewards from all processes and plot the curve
        rewards_queue = mp.Queue()

        print("Running traning on {} processes".format(num_workers))
        processes = []

        for i in range(num_workers):
            idx = int(gpu_list[i])
            process = Process(target=sac_worker, args=(
                idx, sac_trainer, Env, cfg_path, rewards_queue, reward_buffer, replay_buffer, max_episodes, max_steps,
                batch_size, update_itr, explore_steps, action_dim, AUTO_ENTROPY, DETERMINISTIC, out_dir))  # the args contain shared and not shared
            process.daemon = True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep geting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                if len(rewards) == 0:
                    rewards.append(r)
                else:
                    # moving average of episode rewards
                    rewards.append(0.9 * rewards[-1] + 0.1 * r)
            else:
                break

            if len(rewards) % 10 == 0 and len(rewards) > 0:
                plot(rewards, out_dir + '/rewards.png')
                np.save(out_dir + "/rewards", np.array(rewards))

            if len(rewards) >= max_episodes:
                [p.terminate() for p in processes]
                break

        [p.join() for p in processes]  # finished at the same time

    if args.test:
        import time
        # try:
        sac_trainer.load_model(out_dir + "/lbmsac_latest")
        # sac_trainer.load_model(out_dir + "/lbmsac_best")
        
        # except:
        #     print("No model found, please train the model first.", out_dir + "/lbmsac_best")
        #     pass
        sac_trainer.to_cuda()
        env = Env(cfg_path)
        reward_record = []
        energy_record = []

        for eps in range(1):
            state = env.reset()
            episode_reward = 0
            episode_energy = 0

            action_record = []
            t1 = time.time()
            for step in range(max_steps):
                action = sac_trainer.policy_net.get_action(
                    state, deterministic=DETERMINISTIC)
                # action = sac_trainer.policy_net.sample_action()
                # action = np.zeros(action_dim)

                next_state, reward, done, info = env.step(action)

                if done:
                    break
                # print("action: ", action, "reward: ", reward)
                action_record.append(action)

                env.render()

                episode_reward += info['fish_vel_towards_target']
                episode_energy += info['step_energy']
                state = next_state

            t2 = time.time()
            print("FPS: ", max_steps * 40 / (t2 - t1))

            print('Episode: ', eps, '| Episode Reward: ',
                  episode_reward, '| Episode Energy: ', episode_energy)
            reward_record.append(episode_reward * interval)
            energy_record.append(episode_energy)
            # np.save(out_dir + "/action_record.npy", np.array(action_record))

        reward_record = np.array(reward_record)
        energy_record = np.array(energy_record)
        print("Average reward: ", np.mean(reward_record),
              "Std of reward: ", np.std(reward_record))
        print("Average energy: ", np.mean(energy_record),
              "Std of energy: ", np.std(energy_record))


if __name__ == '__main__':
    main()
