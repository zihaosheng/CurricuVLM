import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from imitation.data.types import TrajectoryWithRew

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger

from advgen.adv_generator import AdvGenerator
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from omnisafe_env import convert_gym_to_gymnasium
from saferl_plotter.logger import SafeLogger


def eval_policy(policy, eval_env, adv_generator, eval_episodes=100, record_t=0):
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
        state, done = eval_env.reset(), False
        adv_generator.before_episode(eval_env)
        while not done:
            adv_generator.log_AV_history()
            action, _ = policy.predict(state.reshape(1, -1), deterministic=False)
            state, reward, done, info = eval_env.step(action.flatten())
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
        _episode_distances[ep_num] = info['current_distance']
        _episode_speeds[ep_num] = episode_speed / info['episode_length']
        _episode_overtake[ep_num] = episode_overtake

        adv_generator.after_episode(update_AV_traj=True, mode='eval')

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
        state, done = eval_env.reset(), False
        adv_generator.before_episode(eval_env)
        adv_generator.generate(mode='eval')
        eval_env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_generator.adv_traj)
        while not done:
            action, _ = policy.predict(state.reshape(1, -1), deterministic=False)
            state, reward, done, info = eval_env.step(action.flatten())
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
        _episode_distances[ep_num] = info['current_distance']
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


def load_expert_data():
    rollouts = []
    expert_dir = './expert_data'
    for idx in range(8):
        file_name = 'human_traj_{}_{}.json'.format(idx * 50, idx * 50 + 50)
        with open(os.path.join(expert_dir, file_name), 'r', encoding='utf-8') as f:
            human_traj = json.load(f)
            data = human_traj['data']
            episode_len = human_traj['episode_len']
            for i in range(len(episode_len)):
                start_idx = int(np.sum(episode_len[:i]))
                end_idx = int(np.sum(episode_len[:i + 1]))
                episode_data = data[start_idx:end_idx]
                obs = []
                acts = []
                rewads = []
                for ed in episode_data:
                    obs.append(ed['obs'])
                    acts.append(ed['actions'])
                    rewads.append(ed['rewards'])
                obs.append(episode_data[-1]['new_obs'])
                traj = TrajectoryWithRew(
                    obs=np.array(obs),
                    acts=np.array(acts),
                    rews=np.array(rewads),
                    infos=None,
                    terminal=True,
                )
                rollouts.append(traj)
    return rollouts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MDWaymo")
    parser.add_argument("--eval_freq", default=25000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument('--OV_traj_num', type=int, default=32)  # number of opponent vehicle candidates
    parser.add_argument('--AV_traj_num', type=int,
                        default=5)  # lens of ego traj deque (AV=Autonomous Vehicle is the same as EV(Ego vehcile) in the paper)

    adv_generator = AdvGenerator(parser)
    args = parser.parse_args()

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

    env = WaymoEnv(config=config_test)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    logger = SafeLogger(exp_name='imitation-BC', env_name=args.env, seed=args.seed,
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

    transitions = load_expert_data()
    transitions = rollout.flatten_trajectories(transitions)

    bc_trainer = bc.BC(
        observation_space=convert_gym_to_gymnasium(env.observation_space),
        action_space=convert_gym_to_gymnasium(env.action_space),
        demonstrations=transitions,
        rng=rng,
        device='cpu',
        custom_logger=imit_logger.configure('./tmp', ["csv"])
    )

    for i in range(100):
        bc_trainer.train(n_epochs=1)

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
        eval_episode_overtake_adv = eval_policy(bc_trainer.policy, env, adv_generator, record_t=i)
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
                       eval_episode_overtake_adv], total_steps=i + 1, mode='eval')

        if args.save_model: bc_trainer.policy.save(os.path.join(logger.log_dir, "{}-{}".format("imitation-BC", i)))
