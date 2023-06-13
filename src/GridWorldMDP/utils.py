import numpy as np
from collections import defaultdict, namedtuple
from .value_iteration import value_iteration
from .gridworld import GridWorld

Step = namedtuple('Step','cur_state action next_state reward done')

def draw_path(traj, gw):
    # Step = namedtuple('Step','cur_state action next_state reward done')
    path = ''
    action_dir = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
    for k, step in enumerate(traj):
        cur_state = gw.idx2pos(step.cur_state)
        cur_action = action_dir[step.action]
        next_state = gw.idx2pos(step.next_state)
        path += f's={cur_state}, a={cur_action}, r={step.reward}, s\'={next_state}'
        if k == len(traj)-1:
            path += ''
        else:
            path += ' -> \n'

    return path

def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=None):
    """gatheres expert demonstrations

    inputs:
    gw          Gridworld - the environment
    policy      Nx1 matrix
    n_trajs     int - number of trajectories to generate
    rand_start  bool - randomly picking start position or not
    start_pos   2x1 list - set start position, default [0,0]
    returns:
    trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
    """
    if rand_start:
        assert start_pos is None, 'Start position must be None if randomly picking start position'
    else:
        assert start_pos is not None, 'Start position must be specified if not randomly picking start position'
    
    trajs = []
    for i in range(n_trajs):
        if rand_start:
            # override start_pos
            start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

        episode = []
        gw.reset(start_pos)
        cur_state = start_pos
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
        episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
        # while not is_done:
        for _ in range(len_traj-1):
            cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(next_state)]))
            episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
            if is_done:
                break
        trajs.append(episode)
    return trajs

def init_grid_world(args, coor_rates: list[tuple[tuple[int, int], float]]):
    # init the gridworld
    # rmap_gt is the ground truth for rewards
    print('[INFO] Initialize Grid World')
    rmap_gt = np.zeros([args.height, args.width])
    for coor_r in coor_rates:
        coor, rate = coor_r
        rmap_gt[coor[0], coor[1]] = rate * args.r_max

    gw = GridWorld(rmap_gt, {}, 1 - args.act_random)

    rewards_gt = np.reshape(rmap_gt, args.height*args.width, order='F')
    P_a = gw.get_transition_mat()
    print('[INFO] Getting ground truth values and policy via value teration')
    values_gt, policy_gt = value_iteration(P_a, rewards_gt, args.gamma, error=args.error, deterministic=True)

    return gw, P_a, rewards_gt, values_gt, policy_gt

def visitation_frequency(trajs, n_states):
    freq = np.zeros(n_states, dtype=np.int64)
    for traj in trajs:
        for step in traj:
            freq[step.cur_state] += 1
    return freq