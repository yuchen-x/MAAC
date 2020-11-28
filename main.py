import argparse
import torch
import os
import numpy as np
import random

from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC

from marl_envs.my_env.capture_target import CaptureTarget as CT
from marl_envs.my_env.small_box_pushing import SmallBoxPushing as SBP
from marl_envs.particle_envs.make_env import make_env


def make_parallel_env(config):
    env_id = config.env_id
    n_rollout_threads = config.n_rollout_threads
    seed = config.seed

    if config.env_id.startswith('CT'):
        env_args = {'terminate_step': config.episode_length,
                     'n_target': config.n_target,
                     'n_agent': config.n_agent}
    elif config.env_id.startswith('BP') or config.env_id.startswith('SBP'):
        env_args = {'terminate_step': config.episode_length,
                    'random_init': config.random_init,
                    'small_box_only': config.small_box_only,
                    'terminal_reward_only': config.terminal_reward_only,
                    'big_box_reward': config.big_box_reward,
                    'small_box_reward': config.small_box_reward,
                    'n_agent': config.n_agent}
    else:
        env_args = {'max_epi_steps': config.episode_length,
                    'prey_accel': config.n_target,
                    'prey_max_v': config.prey_max_v,
                    'obs_r': config.obs_r,
                    'obs_resolution': config.obs_resolution,
                    'flick_p': config.flick_p,
                    'enable_boundary': config.enable_boundary,
                    'benchmark': config.benchmark,
                    'discrete_mul': config.discrete_mul,
                    'config_name': config.config_name}

    def get_env_fn(rank):
        def init_env():
            if env_id.startswith('CT'):
                env = CT(grid_dim=tuple(config.grid_dim), **env_args)
            elif env_id.startswith('SBP'):
                env = SBP(tuple(config.grid_dim), **env_args)
            else:
                env = make_env(env_id, 
                               discrete_action=True, 
                               discrete_action_input=True, 
                               **env_args)
                env.seed(seed + rank)
            np.random.seed(seed + rank)
            random.seed(seed + rank)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    env = make_parallel_env(config)

    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        torch_H = [[None] for _ in range(obs.shape[1])]
        import ipdb
        ipdb.set_trace()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False).unsqueeze(1)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions, torch_H = model.step(torch_obs, H=torch_H, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, _  = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=2, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=1, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=64, type=int)
    parser.add_argument("--critic_hidden_dim", default=64, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--seed", default=0, type=int)
    # env args
    parser.add_argument('--grid_dim', nargs=2, default=[4,4], type=int)
    parser.add_argument("--n_target", default=1, type=int)
    parser.add_argument("--n_agent", default=2, type=int)
    parser.add_argument("--small_box_only", action='store_true')
    parser.add_argument("--terminal_reward_only", action='store_true')
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--small_box_reward", default=100.0, type=float)
    parser.add_argument("--big_box_reward", default=100.0, type=float)
    # particle envs args
    parser.add_argument("--prey_accel", default=4.0, type=float)
    parser.add_argument("--prey_max_v", default=1.3, type=float)
    parser.add_argument("--flick_p", default=0.0, type=float)
    parser.add_argument("--obs_r", default=2.0, type=float)
    parser.add_argument("--enable_boundary", action='store_true')
    parser.add_argument("--discrete_mul", default=1, type=int)
    parser.add_argument("--config_name", default="antipodal", type=str)
    parser.add_argument("--benchmark", action='store_true')
    parser.add_argument("--obs_resolution", default=8, type=int)

    config = parser.parse_args()

    run(config)
