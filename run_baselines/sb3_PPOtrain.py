import numpy as np
import torch
import argparse
import os

from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor

from sb3.ppo.ppo import PPO

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from advgen.adv_generator import AdvGenerator

from saferl_plotter.logger import SafeLogger


def eval_policy(policy, eval_env, adv_generator, eval_episodes=100):
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
            action = policy.select_action(obs=state.reshape(1, -1))
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
            action = policy.select_action(obs=state.reshape(1, -1))
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
    parser.add_argument("--start_timesteps", default=10000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25000, type=int)  # How often (time steps) we evaluate
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

    file_name = "{}-PPO".format(args.mode)
    logger = SafeLogger(exp_name="{}-PPO".format(args.mode), env_name=args.env, seed=args.seed,
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

    extra_args = dict(mode="top_down", film_size=(2200, 2200))
    # Set seeds
    env = WaymoEnv(config=config_train)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ########################################
    policy = PPO('MlpPolicy', env, verbose=0, seed=0, device='cuda')
    new_logger = configure(logger.log_dir, ["csv"])
    policy.set_logger(new_logger)
    # setup learn
    if policy.action_noise is not None:
        policy.action_noise.reset()

    if args.reset_num_timesteps:
        policy.num_timesteps = 0
        policy._episode_num = 0
    else:
        # Make sure training timesteps are ahead of the internal counter
        args.total_timesteps += policy.num_timesteps
    policy._total_timesteps = args.total_timesteps
    policy._num_timesteps_at_start = policy.num_timesteps

    # Avoid resetting the environment when calling ``.learn()`` consecutive times
    if args.reset_num_timesteps or policy._last_obs is None:
        assert policy.env is not None
        # pytype: disable=annotation-type-mismatch
        policy._last_obs = policy.env.reset()  # type: ignore[assignment]
        # pytype: enable=annotation-type-mismatch
        policy._last_episode_starts = np.ones((policy.env.num_envs,), dtype=bool)
        # Retrieve unnormalized observation for saving into the buffer
        if policy._vec_normalize_env is not None:
            policy._last_original_obs = policy._vec_normalize_env.get_original_obs()
    # ########################################

    new_obs = policy._last_obs
    adv_generator.before_episode(env)
    # env.render(**extra_args)
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0

    last_eval_step = 0

    num_advgen = 0

    updates = 0

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

    while policy.num_timesteps < int(args.total_timesteps):
        # ##############################################################################
        policy.policy.set_training_mode(False)

        n_steps = 0
        policy.rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if policy.use_sde:
            policy.policy.reset_noise()

        while n_steps < policy.n_steps:
            episode_timesteps += 1
            adv_generator.log_AV_history()

            if policy.use_sde and policy.sde_sample_freq > 0 and n_steps % policy.sde_sample_freq == 0:
                # Sample a new noise matrix
                policy.policy.reset_noise()

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(policy._last_obs, policy.device)
                actions, values, log_probs = policy.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            # if isinstance(policy.action_space, spaces.Box):
            clipped_actions = np.clip(actions, policy.action_space.low, policy.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions.flatten())

            policy.num_timesteps += 1

            n_steps += 1

            policy.rollout_buffer.add(
                policy._last_obs,  # type: ignore[arg-type]
                actions,
                np.array([rewards]),
                policy._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            policy._last_obs = new_obs.reshape(1, -1)  # type: ignore[assignment]
            policy._last_episode_starts = np.array([dones])

            if env.vehicle.crash_vehicle and not is_collision:
                collision_t = policy.num_timesteps
                is_collision = True

            # env.render(**extra_args)

            episode_reward += rewards
            episode_cost += infos['cost']
            episode_overtake += infos['overtake_vehicle_num']
            episode_speed += infos['velocity']

            if dones:
                ep_num = episode_num % args.train_eval_episode
                _route_completion[ep_num] = infos['route_completion']
                _crash_rates[ep_num] = float(is_collision)
                _out_of_road_rates[ep_num] = float(infos['out_of_road'])
                _success_rates[ep_num] = float(infos['arrive_dest'])
                _episode_rewards[ep_num] = infos['episode_reward']
                _episode_costs[ep_num] = episode_cost
                _episode_distances[ep_num] = infos['current_distance']
                _episode_speeds[ep_num] = episode_speed / infos['episode_length']
                _episode_overtake[ep_num] = episode_overtake

                print('#' * 20)
                print(
                    f"Total T: {policy.num_timesteps + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
                print(
                    f"arrive destination: {infos['arrive_dest']} , route_completion: {infos['route_completion']}, out of road:{infos['out_of_road']}, crash: {env.vehicle.crash_vehicle}  ")
                # print('='*30)
                # print(infos)

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
                                   avg_episode_overtake_normal, ], total_steps=policy.num_timesteps + 1, mode='train')

                # Evaluate episode
                if policy.num_timesteps - last_eval_step > args.eval_freq:
                    last_eval_step = policy.num_timesteps
                    env.close()
                    eval_env = WaymoEnv(config=config_test)
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
                    eval_episode_overtake_adv = eval_policy(policy, eval_env,
                                                            adv_generator)
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
                                   eval_episode_overtake_adv], total_steps=policy.num_timesteps + 1, mode='eval')

                    env = WaymoEnv(config=config_train)

                    if args.save_model: policy.save(os.path.join(logger.log_dir, "{}-{}".format(file_name, policy.num_timesteps)))

                # Reset environment
                new_obs, dones = env.reset(), False
                adv_generator.before_episode(env)

                if args.mode == 'cat' and np.random.random() > max(
                    1 - (2 * policy.num_timesteps / args.total_timesteps) * (1 - args.min_prob),
                    args.min_prob):
                    print('ADVGEN')
                    adv_generator.generate()
                    num_advgen += 1
                    print('# is {}'.format(num_advgen))
                else:
                    print('NORMAL')

                env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent, adv_generator.adv_traj)
                episode_reward = 0
                episode_cost = 0
                episode_timesteps = 0
                episode_num += 1
                is_collision = False
                episode_overtake = 0
                episode_speed = 0

        with torch.no_grad():
            # Compute value for the last timestep
            values = policy.policy.predict_values(obs_as_tensor(new_obs.reshape(1, -1), policy.device))  # type: ignore[arg-type]

        policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=np.array([dones]))

        policy._update_current_progress_remaining(policy.num_timesteps, args.total_timesteps)

        policy.train()

        # ##############################################################################


    logger.close()
