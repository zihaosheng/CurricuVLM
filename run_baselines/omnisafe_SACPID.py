import numpy as np
import omnisafe
import pandas as pd
import torch
import argparse
import os

from advgen.adv_generator import AdvGenerator

from saferl_plotter.logger import SafeLogger


def eval_policy(agent, eval_env, adv_generator, eval_episodes=100, record_t=0):
    extra_args = dict(mode="top_down", film_size=(2200, 2200))
    _route_completion = [0.] * eval_episodes
    _crash_rates = [0.] * eval_episodes
    _out_of_road_rates = [0.] * eval_episodes
    _success_rates = [0.] * eval_episodes
    _episode_rewards = [0.] * eval_episodes
    _episode_costs = [0.] * eval_episodes
    _episode_distances = [0.] * eval_episodes
    _episode_speeds = [0.] * eval_episodes
    _episode_overtake = [0.] * eval_episodes

    print("=" * 40)
    print("Eval on NORMAL")
    print("=" * 40)

    for ep_num in range(eval_episodes):
        episode_cost = 0
        episode_overtake = 0
        episode_speed = 0
        _current_obs, info = eval_env.reset(seed=ep_num+400)
        done = False
        adv_generator.before_episode(eval_env)

        # eval_env.render(**extra_args)
        # eval_env.engine.top_down_renderer.set_adv(adv_generator.adv_agent)
        while not done:
            adv_generator.log_AV_history()
            act = agent.agent._actor_critic.step(_current_obs, deterministic=False)
            _current_obs, reward, cost, terminated, truncated, info = eval_env.step(act)
            # eval_env.render(**extra_args)

            done = torch.logical_or(terminated, truncated)

            episode_cost += info['cost']
            if 'final_info' in info:
                episode_overtake += info['final_info']['overtake_vehicle_num']
                episode_speed += info['final_info']['velocity']
            else:
                episode_overtake += info['overtake_vehicle_num']
                episode_speed += info['velocity']


        print('#' * 20)
        print(
            f"arrive destination: {info['arrive_dest']} , route_completion: {info['route_completion']}, out of road:{info['out_of_road']}, crash: {eval_env.vehicle.crash_vehicle}   ")

        _route_completion[ep_num] = info['route_completion']
        crash = 1 if eval_env.vehicle.crash_vehicle else 0
        _crash_rates[ep_num] = crash
        _out_of_road_rates[ep_num] = float(info['out_of_road'])
        _success_rates[ep_num] = float(info['arrive_dest'])
        _episode_rewards[ep_num] = info['episode_reward']
        _episode_costs[ep_num] = episode_cost
        _episode_distances[ep_num] = info['track_length'] * _route_completion[ep_num]
        _episode_speeds[ep_num] = episode_speed / info['episode_length']
        _episode_overtake[ep_num] = episode_overtake

        eval_env.last_seed = eval_env.current_seed

        adv_generator.after_episode(update_AV_traj=True, mode='eval', is_safe=True)

    df = pd.DataFrame({
        'success_rates': _success_rates,
        'crash_rates': _crash_rates,
        'out_of_road_rates': _out_of_road_rates
    })
    df.to_csv(os.path.join(logger.log_dir, 'normal_{}_rates_data.csv'.format(record_t)), index=False)

    avg_route_completion_normal = sum(_route_completion) / eval_episodes
    avg_crash_rate_normal = sum(_crash_rates) / eval_episodes
    avg_out_of_road_rate_normal = sum(_out_of_road_rates) / eval_episodes
    avg_success_rate_normal = sum(_success_rates) / eval_episodes
    avg_episode_reward_normal = sum(_episode_rewards) / eval_episodes
    avg_episode_cost_normal = sum(_episode_costs) / eval_episodes
    avg_episode_distance_normal = sum(_episode_distances) / eval_episodes
    avg_episode_speed_normal = sum(_episode_speeds) / eval_episodes
    avg_episode_overtake_normal = sum(_episode_overtake) / eval_episodes

    print("=" * 40)
    print("Eval on ADV")
    print("=" * 40)

    _route_completion = [0.] * eval_episodes
    _crash_rates = [0.] * eval_episodes
    _out_of_road_rates = [0.] * eval_episodes
    _success_rates = [0.] * eval_episodes
    _episode_rewards = [0.] * eval_episodes
    _episode_costs = [0.] * eval_episodes
    _episode_distances = [0.] * eval_episodes
    _episode_speeds = [0.] * eval_episodes
    _episode_overtake = [0.] * eval_episodes

    for ep_num in range(eval_episodes):
        episode_cost = 0
        episode_overtake = 0
        episode_speed = 0
        _current_obs, info = eval_env.reset(seed=ep_num+400)
        done = False
        adv_generator.before_episode(eval_env)

        # eval_env.render(**extra_args)
        # eval_env.engine.top_down_renderer.set_adv(adv_generator.adv_agent)
        adv_generator.generate(mode='eval')
        eval_env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_generator.adv_traj)
        while not done:
            adv_generator.log_AV_history()
            act = agent.agent._actor_critic.step(_current_obs, deterministic=False)
            _current_obs, reward, cost, terminated, truncated, info = eval_env.step(act)
            # eval_env.render(**extra_args)

            done = torch.logical_or(terminated, truncated)

            episode_cost += info['cost']
            episode_overtake += info['overtake_vehicle_num']
            episode_speed += info['velocity']

        print('#' * 20)
        print(
            f"arrive destination: {info['arrive_dest']} , route_completion: {info['route_completion']}, out of road:{info['out_of_road']}, crash: {eval_env.vehicle.crash_vehicle}   ")

        _route_completion[ep_num] = info['route_completion']
        crash = 1 if eval_env.vehicle.crash_vehicle else 0
        _crash_rates[ep_num] = crash
        _out_of_road_rates[ep_num] = float(info['out_of_road'])
        _success_rates[ep_num] = float(info['arrive_dest'])
        _episode_rewards[ep_num] = info['episode_reward']
        _episode_costs[ep_num] = episode_cost
        _episode_distances[ep_num] = info['track_length'] * _route_completion[ep_num]
        _episode_speeds[ep_num] = episode_speed / info['episode_length']
        _episode_overtake[ep_num] = episode_overtake

    df = pd.DataFrame({
        'success_rates': _success_rates,
        'crash_rates': _crash_rates,
        'out_of_road_rates': _out_of_road_rates
    })
    df.to_csv(os.path.join(logger.log_dir, 'adv_{}_rates_data.csv'.format(record_t)), index=False)

    avg_route_completion_adv = sum(_route_completion) / eval_episodes
    avg_crash_rate_adv = sum(_crash_rates) / eval_episodes
    avg_out_of_road_rate_adv = sum(_out_of_road_rates) / eval_episodes
    avg_success_rate_adv = sum(_success_rates) / eval_episodes
    avg_episode_reward_adv = sum(_episode_rewards) / eval_episodes
    avg_episode_cost_adv = sum(_episode_costs) / eval_episodes
    avg_episode_distance_adv = sum(_episode_distances) / eval_episodes
    avg_episode_speed_adv = sum(_episode_speeds) / eval_episodes
    avg_episode_overtake_adv = sum(_episode_overtake) / eval_episodes

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: Reward_normal {avg_route_completion_normal:.3f} Cost_normal {avg_crash_rate_normal: .3f} Reward_adv {avg_route_completion_adv:.3f} Cost_adv {avg_crash_rate_adv: .3f}")
    print("---------------------------------------")
    return avg_route_completion_normal, avg_crash_rate_normal, avg_out_of_road_rate_normal, avg_success_rate_normal, avg_episode_reward_normal, avg_episode_cost_normal, avg_episode_distance_normal, avg_episode_speed_normal, avg_episode_overtake_normal, \
           avg_route_completion_adv, avg_crash_rate_adv, avg_out_of_road_rate_adv, avg_success_rate_adv, avg_episode_reward_adv, avg_episode_cost_adv, avg_episode_distance_adv, avg_episode_speed_adv, avg_episode_overtake_adv


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MDWaymo")
    parser.add_argument("--start_timesteps", default=10_000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25_000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--train_eval_episode", default=100,
                        type=int)  # How often (episodes) we check the training performance
    parser.add_argument("--total_timesteps", default=1_000_000, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--reset_num_timesteps", action="store_false")  # reset_num_timesteps
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument('--OV_traj_num', type=int, default=32)  # number of opponent vehicle candidates
    parser.add_argument('--AV_traj_num', type=int,
                        default=5)  # lens of ego traj deque (AV=Autonomous Vehicle is the same as EV(Ego vehcile) in the paper)
    parser.add_argument('--min_prob', type=float, default=0.1)  # The min probability of using raw data in ADV mode
    parser.add_argument('--mode', choices=['replay', 'cat'], \
                        help='Choose a mode (replay, cat)', default='cat')

    adv_generator = AdvGenerator(parser)
    args = parser.parse_args()

    file_name = "{}-OmniSafe-SACPID".format(args.mode)
    logger = SafeLogger(exp_name="{}-OmniSafe-SACPID".format(args.mode), env_name=args.env, seed=args.seed,
                        fieldnames=['route_completion_normal',
                                    'crash_rate_normal',
                                    'out_of_road_rate_normal',
                                    'success_rate_normal',
                                    'episode_reward_normal',
                                    'episode_cost_normal',
                                    'episode_distance_normal',
                                    'episode_speed_normal',
                                    'episode_overtake_normal',
                                    'route_completion_adv',
                                    'crash_rate_adv',
                                    'out_of_road_rate_adv',
                                    'success_rate_adv',
                                    'episode_reward_adv',
                                    'episode_cost_adv',
                                    'episode_distance_adv',
                                    'episode_speed_adv',
                                    'episode_overtake_adv'])

    config_train = dict(
        data_directory=os.path.join(os.path.dirname(__file__), "../raw_scenes_500"),
        start_scenario_index=0,
        num_scenarios=400,
        sequential_seed=False,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        no_static_vehicles=True,
        reactive_traffic=False,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    )

    config_test = dict(
        data_directory=os.path.join(os.path.dirname(__file__), "../raw_scenes_500"),
        start_scenario_index=400,
        num_scenarios=100,
        crash_vehicle_done=True,
        sequential_seed=True,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        no_static_vehicles=True,
        reactive_traffic=False,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    )
    # env = ScenarioEnv(config=config_train)
    # ######################################################
    from omnisafe_env import SafetyMetaDriveEnv

    env_id = 'SafeMetaDrive'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 1_000_000,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 20000,
            'obs_normalize': True,
        },
        'model_cfgs': {
            'actor': {
                'lr': 0.0001
            },
            'critic': {
                'lr': 0.0001
            },
        },
        'lagrange_cfgs': {
            'cost_limit': 1.0
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
        },
        '"../raw_scenes_500"': {
            'meta_drive_config': config_train
        }
    }
    test_cfgs = {
        '"../raw_scenes_500"': {
            'meta_drive_config': config_test
        }
    }
    from omnisafe.algorithms.off_policy.sac import SAC

    # policy = SAC(env_id=env_id, cfgs=)
    agent = omnisafe.Agent(algo='SACPID', env_id=env_id, custom_cfgs=custom_cfgs)
    print(agent.cfgs)
    # _current_obs = env.reset()
    _current_obs = agent.agent._env._current_obs

    adv_generator.before_episode(agent.agent._env._env)

    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0

    last_eval_step = 0

    num_advgen = 0

    _route_completion = [0.] * args.train_eval_episode
    _crash_rates = [0.] * args.train_eval_episode
    _out_of_road_rates = [0.] * args.train_eval_episode
    _success_rates = [0.] * args.train_eval_episode
    _episode_rewards = [0.] * args.train_eval_episode
    _episode_costs = [0.] * args.train_eval_episode
    _episode_distances = [0.] * args.train_eval_episode
    _episode_speeds = [0.] * args.train_eval_episode
    _episode_overtake = [0.] * args.train_eval_episode

    episode_overtake = 0
    episode_speed = 0

    is_collision = False

    t = 0

    for epoch in range(agent.agent._epochs):
        agent.agent._epoch = epoch
        for sample_step in range(
            epoch * agent.agent._samples_per_epoch,
            (epoch + 1) * agent.agent._samples_per_epoch,
        ):
            step = sample_step * agent.agent._update_cycle * agent.agent._cfgs.train_cfgs.vector_env_nums

            # set noise for exploration
            if agent.agent._cfgs.algo_cfgs.use_exploration_noise:
                agent.agent._actor_critic.actor.noise = agent.agent._cfgs.algo_cfgs.exploration_noise

            for _ in range(agent.agent._update_cycle):
                t += 1
                if step <= agent.agent._cfgs.algo_cfgs.start_learning_steps:
                    act = (torch.rand(agent.agent._env.action_space.shape) * 2 - 1).unsqueeze(0).to(agent.agent._device)  # type: ignore
                else:
                    act = agent.agent._actor_critic.step(_current_obs, deterministic=False)
                # next_obs, reward, cost, terminated, truncated, info = env.step(act.detach().cpu().numpy())
                next_obs, reward, cost, terminated, truncated, info = agent.agent._env.step(act)
                # print(act)

                episode_timesteps += 1

                adv_generator.log_AV_history()

                # self._log_value(reward=reward, cost=cost, info=info)
                real_next_obs = next_obs.clone()

                if agent.agent._env._env.vehicle.crash_vehicle and not is_collision:
                    collision_t = t
                    is_collision = True

                if 'original_reward' in info:
                    episode_reward += info['original_reward'].detach().cpu().numpy()[0]
                else:
                    episode_reward += reward.detach().cpu.numpy()[0]
                if 'original_cost' in info:
                    episode_cost += info['original_cost'].detach().cpu().numpy()[0]
                else:
                    episode_cost += cost.detach().cpu.numpy()[0]
                if 'final_info' in info:
                    episode_overtake += info['final_info']['overtake_vehicle_num']
                    episode_speed += info['final_info']['velocity']
                else:
                    episode_overtake += info['overtake_vehicle_num']
                    episode_speed += info['velocity']

                for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                    if done:
                        if 'final_observation' in info:
                            real_next_obs[idx] = info['final_observation'][idx]

                        ep_num = episode_num % args.train_eval_episode
                        _route_completion[ep_num] = info['final_info']['route_completion']
                        _crash_rates[ep_num] = float(is_collision)
                        _out_of_road_rates[ep_num] = float(info['final_info']['out_of_road'])
                        _success_rates[ep_num] = float(info['final_info']['arrive_dest'])
                        _episode_rewards[ep_num] = info['final_info']['episode_reward']
                        _episode_costs[ep_num] = episode_cost
                        _episode_distances[ep_num] = info['final_info']['track_length'] * _route_completion[ep_num]
                        _episode_speeds[ep_num] = episode_speed / info['final_info']['episode_length']
                        _episode_overtake[ep_num] = episode_overtake

                        adv_generator.after_episode(update_AV_traj=args.mode == 'cat', is_safe=True)

                        print('#' * 20)
                        print(
                            f"Total T: {t} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
                        print(
                            f"arrive destination: {info['final_info']['arrive_dest']} , route_completion: {info['final_info']['route_completion']}, out of road:{info['final_info']['out_of_road']}, crash: {agent.agent._env._env.vehicle.crash_vehicle}  ")
                        # print('='*30)
                        # print(info)

                        if (episode_num + 1) % args.train_eval_episode == 0:
                            avg_route_completion_normal = sum(_route_completion) / args.train_eval_episode
                            avg_crash_rate_normal = sum(_crash_rates) / args.train_eval_episode
                            avg_out_of_road_rate_normal = sum(_out_of_road_rates) / args.train_eval_episode
                            avg_success_rate_normal = sum(_success_rates) / args.train_eval_episode
                            avg_episode_reward_normal = sum(_episode_rewards) / args.train_eval_episode
                            avg_episode_cost_normal = sum(_episode_costs) / args.train_eval_episode
                            avg_episode_distance_normal = sum(_episode_distances) / args.train_eval_episode
                            avg_episode_speed_normal = sum(_episode_speeds) / args.train_eval_episode
                            avg_episode_overtake_normal = sum(_episode_overtake) / args.train_eval_episode

                            logger.update([avg_route_completion_normal,
                                           avg_crash_rate_normal,
                                           avg_out_of_road_rate_normal,
                                           avg_success_rate_normal,
                                           avg_episode_reward_normal,
                                           avg_episode_cost_normal,
                                           avg_episode_distance_normal,
                                           avg_episode_speed_normal,
                                           avg_episode_overtake_normal,
                                           avg_route_completion_normal,
                                           avg_crash_rate_normal,
                                           avg_out_of_road_rate_normal,
                                           avg_success_rate_normal,
                                           avg_episode_reward_normal,
                                           avg_episode_cost_normal,
                                           avg_episode_distance_normal,
                                           avg_episode_speed_normal,
                                           avg_episode_overtake_normal, ], total_steps=t, mode='train')

                        # Evaluate episode
                        if t - last_eval_step > args.eval_freq:
                            last_eval_step = t
                            agent.agent._env.close()
                            eval_env = SafetyMetaDriveEnv(env_id=env_id, **test_cfgs['"../raw_scenes_500"'])
                            eval_route_completion_normal, \
                            eval_crash_rate_normal, \
                            eval_out_of_road_rate_normal, \
                            eval_success_rate_normal, \
                            eval_episode_reward_normal, \
                            eval_episode_cost_normal, \
                            eval_episode_distance_normal, \
                            eval_episode_speed_normal, \
                            eval_episode_overtake_normal, \
                            eval_route_completion_adv, \
                            eval_crash_rate_adv, \
                            eval_out_of_road_rate_adv, \
                            eval_success_rate_adv, \
                            eval_episode_reward_adv, \
                            eval_episode_cost_adv, \
                            eval_episode_distance_adv, \
                            eval_episode_speed_adv, \
                            eval_episode_overtake_adv = eval_policy(agent, eval_env,
                                                                    adv_generator, record_t=t+1)
                            eval_env.close()
                            logger.update([eval_route_completion_normal,
                                           eval_crash_rate_normal,
                                           eval_out_of_road_rate_normal,
                                           eval_success_rate_normal,
                                           eval_episode_reward_normal,
                                           eval_episode_cost_normal,
                                           eval_episode_distance_normal,
                                           eval_episode_speed_normal,
                                           eval_episode_overtake_normal,
                                           eval_route_completion_adv,
                                           eval_crash_rate_adv,
                                           eval_out_of_road_rate_adv,
                                           eval_success_rate_adv,
                                           eval_episode_reward_adv,
                                           eval_episode_cost_adv,
                                           eval_episode_distance_adv,
                                           eval_episode_speed_adv,
                                           eval_episode_overtake_adv], total_steps=t + 1, mode='eval')

                            agent.agent._init_env()
                            next_obs = agent.agent._env._current_obs

                        adv_generator.before_episode(agent.agent._env._env)
                        if args.mode == 'cat' and np.random.random() > max(1 - (2 * t / args.total_timesteps) * (1 - args.min_prob),
                                                                           args.min_prob):
                            print('ADVGEN')
                            adv_generator.generate()
                            num_advgen += 1
                            print('# is {}'.format(num_advgen))
                        else:
                            print('NORMAL')

                        episode_reward = 0
                        episode_cost = 0
                        episode_timesteps = 0
                        episode_num += 1
                        is_collision = False
                        episode_overtake = 0
                        episode_speed = 0


                agent.agent._buf.store(
                    obs=_current_obs,
                    act=act,
                    reward=reward,
                    cost=cost,
                    done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                    next_obs=real_next_obs,
                )

                _current_obs = next_obs


            if step > agent.agent._cfgs.algo_cfgs.start_learning_steps:
                agent.agent._update()


    logger.close()
