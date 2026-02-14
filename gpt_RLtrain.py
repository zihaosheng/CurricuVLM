import uuid
import numpy as np
import torch
import argparse
import os

from PIL import Image

from advgen.gpt_utils import query_gpt_recommendation
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from advgen.adv_generator_gpt import AdvGenerator

from saferl_algo import TD3
from saferl_algo.utils import eval_policy, ReplayBuffer
from saferl_plotter.logger import SafeLogger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="MDWaymo")
    parser.add_argument("--start_timesteps", default=10000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=25000, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--train_eval_episode", default=100,
                        type=int)  # How often (episodes) we check the training performance
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument('--OV_traj_num', type=int, default=32)  # number of opponent vehicle candidates
    parser.add_argument('--AV_traj_num', type=int,
                        default=5)  # lens of ego traj deque (AV=Autonomous Vehicle is the same as EV(Ego vehcile) in the paper)
    parser.add_argument('--min_prob', type=float, default=0.1)  # The min probability of using raw data in ADV mode
    parser.add_argument('--mode', choices=['replay', 'cat', 'gpt'], \
                        help='Choose a mode (replay, cat, gpt)', default='gpt')
    parser.add_argument("--gpt_eval_freq", default=15000, type=int)  # How often (time steps) LLM evaluates
    parser.add_argument("--openai_key", default="")

    adv_generator = AdvGenerator(parser)
    args = parser.parse_args()

    file_name = args.mode
    logger = SafeLogger(exp_name=file_name, env_name=args.env, seed=args.seed,
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

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    config_train = dict(
        data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
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
        use_render=True,
        camera_height=25.2,
        camera_dist=7.5,
    )

    config_test = dict(
        data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
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

    extra_args = dict(mode="rgb_array", film_size=(2200, 2200))
    # Set seeds
    env = WaymoEnv(config=config_train)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    env.render(**extra_args)
    adv_generator.before_episode(env)
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0

    last_eval_step = 0
    last_gpt_eval_step = 0

    num_advgen = 0

    success_scenarios = []
    is_collision = False
    collision_t = None
    collision_figs_dir = logger.log_dir + 'collision_figs/'
    gpt_evaluation_num = 0
    current_collision_figs_dir = collision_figs_dir + str(gpt_evaluation_num)

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

    if not os.path.exists(current_collision_figs_dir):
        os.makedirs(current_collision_figs_dir)

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        adv_generator.log_AV_history()

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, info = env.step(action)
        done_bool = float(done)
        rendered = env.render(**extra_args)

        if env.vehicle.crash_vehicle and not is_collision:
            image = Image.fromarray(rendered.astype(np.uint8))
            cs_path = os.path.join(current_collision_figs_dir, "{}.png".format(uuid.uuid4().hex))
            image.save(cs_path)
            adv_generator.collision_figs.append(cs_path)
            collision_t = t
            is_collision = True
            adv_generator.collision_agent = env.vehicle.crash_vehicle

        if collision_t is not None and t == collision_t + 1:
            collision_t = None

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        episode_cost += info['cost']
        episode_overtake += info['overtake_vehicle_num']
        episode_speed += info['velocity']

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            ep_num = episode_num % args.train_eval_episode
            _route_completion[ep_num] = info['route_completion']
            _crash_rates[ep_num] = float(is_collision)
            _out_of_road_rates[ep_num] = float(info['out_of_road'])
            _success_rates[ep_num] = float(info['arrive_dest'])
            _episode_rewards[ep_num] = info['episode_reward']
            _episode_costs[ep_num] = episode_cost
            _episode_distances[ep_num] = info['current_distance']
            _episode_speeds[ep_num] = episode_speed / info['episode_length']
            _episode_overtake[ep_num] = episode_overtake

            adv_generator.after_episode(update_AV_traj=args.mode != 'replay')

            print('#' * 20)
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            print(
                f"arrive destination: {info['arrive_dest']} , route_completion: {info['route_completion']}, out of road:{info['out_of_road']}, crash: {env.vehicle.crash_vehicle}   ")

            is_success = info['arrive_dest'] and (not env.vehicle.crash_vehicle)  # and (not info['out_of_road'])
            if is_success:
                print("Scenario {} is succeed!".format(env.current_seed))
                success_scenarios.append(env.current_seed)

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
                               avg_episode_overtake_normal, ], total_steps=t + 1, mode='train')

            # GPT evaluates and analyzes current AV agent performance
            if len(adv_generator.collision_figs) == adv_generator.collision_figs.maxlen:
                gpt_evaluation_num += 1
                last_gpt_eval_step = t

                gpt_response = query_gpt_recommendation(args.openai_key, fig_dir=current_collision_figs_dir,
                                                        adv_generator=adv_generator)
                adv_generator.gpt_response = gpt_response

                current_collision_figs_dir = collision_figs_dir + str(gpt_evaluation_num)
                if not os.path.exists(current_collision_figs_dir):
                    os.makedirs(current_collision_figs_dir)

                adv_generator.collision_figs.clear()

            # Evaluate episode
            if t - last_eval_step > args.eval_freq:
                last_eval_step = t
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
                               eval_episode_overtake_adv], total_steps=t + 1, mode='eval')

                env = WaymoEnv(config=config_train)

                if args.save_model: policy.save("./{}/{}-{}".format(logger.log_dir, file_name, t))

            # Reset environment
            state, done = env.reset(), False
            adv_generator.before_episode(env)

            adv_prob = np.random.random() > max(1 - (2 * t / args.max_timesteps) * (1 - args.min_prob), args.min_prob)
            if args.mode == 'gpt' and adv_prob:
                print('ADVGEN+LLM')
                adv_generator.generate()
                num_advgen += 1
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

    logger.close()
