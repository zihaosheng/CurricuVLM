import json
import os
import traceback

import numpy as np
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.manual_control_policy import ManualControlPolicy


def process_info(info):
    ret = {}
    for k, v in info.items():
        # filter float 32
        if k != "raw_action":
            ret[k] = v
        if k == "action":
            ret[k] = v.tolist()
    return ret


if __name__ == '__main__':
    """
    Data = Tuple[o, a, d, r, i]
    """
    start_idx = 0
    episode_idx = start_idx
    num = int(50) + start_idx
    pool = []

    cfgs = dict(
        data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
        manual_control=True,
        use_render=True,
        agent_policy=ManualControlPolicy,
        start_scenario_index=0,
        num_scenarios=500,
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

    env = WaymoEnv(cfgs)
    episode_success = []
    episode_reward = []
    episode_cost = []
    episode_overtake = []
    episode_speed = []
    episode_distance = []

    ignored_episodes = [33, 179, 458]

    total_reward = 0
    total_cost = 0
    avg_speed = 0
    total_overtake = 0

    obs = env.reset(force_seed=episode_idx)

    episode_len = []
    last = 0
    while episode_idx < num:
        last += 1
        new_obs, reward, done, info = env.step([0, 0, 1, 0, 0])
        env.render(
            text={
                "seed": env.current_seed
            }
        )
        action = info["raw_action"]
        total_cost += info["cost"]
        avg_speed += info['velocity']
        total_overtake += info['overtake_vehicle_num']

        pool.append({
            "obs": obs.tolist(), 
            "actions": list(action), 
            "new_obs": new_obs.tolist(), 
            "dones": done,
            "rewards": reward, 
            "infos": process_info(info)
        })
        # print(info['raw_action'], info['action'])

        obs = new_obs
        total_reward += reward
        if done:
            if not info["arrive_dest"] and episode_idx not in ignored_episodes:
                pool = pool[:-last]
            else:
                episode_success.append(1)
                episode_idx += 1
                avg_speed /= last
                episode_reward.append(total_reward)
                episode_cost.append(total_cost)
                episode_speed.append(avg_speed)
                episode_overtake.append(total_overtake)
                episode_distance.append(info['current_distance'])
                episode_len.append(last)
                print(
                    'reset:', episode_idx,
                    "this_episode_len:", last,
                    "total_success_rate: {:.2f}".format(np.mean(episode_success)),
                    "mean_episode_reward: {:.2f}({:.2f})".format(np.mean(episode_reward), np.std(episode_reward)),
                    "mean_episode_cost: {:.2f}({:.2f})".format(np.mean(episode_cost), np.std(episode_cost)),
                    "mean_episode_speed: {:.2f}({:.2f})".format(np.mean(episode_speed), np.std(episode_speed)),
                    "mean_episode_overtake: {:.2f}({:.2f})".format(np.mean(episode_overtake), np.std(episode_overtake)),
                    "mean_episode_distance: {:.2f}({:.2f})".format(np.mean(episode_distance), np.std(episode_distance)),
                )

            print(len(pool))
            if episode_idx < num:
                obs = env.reset(force_seed=episode_idx)
                total_reward = 0
                total_cost = 0
                avg_speed = 0
                total_overtake = 0
                last = 0
                print('finish {}'.format(episode_idx))

    data_set = {"data": pool, "episode_reward": episode_reward, "episode_cost": episode_cost,
                "success_rate": np.mean(episode_success), "episode_len": episode_len,
                "episode_speed": episode_speed, "episode_overtake": episode_overtake,
                "episode_distance": episode_distance}
    try:
        with open('human_traj_{}_{}.json'.format(start_idx, num), 'w') as f:
            json.dump(data_set, f)
            print("total episode_len: ", np.sum(episode_len))
    except Exception as e:
        # print(data_set)
        print(e)
        traceback.print_exc()  