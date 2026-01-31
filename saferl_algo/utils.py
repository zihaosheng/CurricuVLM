import numpy as np
import torch


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
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
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
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
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


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()

            if num_cuda_devices > 1:
                self.device = torch.device("cuda:1")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # TODO 加上importance sampling
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class ReplayBufferLag(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, cost, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # TODO 加上importance sampling
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
