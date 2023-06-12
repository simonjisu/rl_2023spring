import numpy as np
from collections import defaultdict, namedtuple
from .value_iteration import value_iteration
from .objectworld import Objectworld
Step = namedtuple('Step','cur_state action next_state reward done')

def draw_path(traj, ow):
    # Step = namedtuple('Step','cur_state action next_state reward done')
    path = ''
    action_dir = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
    for k, step in enumerate(traj):
        cur_state = ow.idx2pos(step.cur_state)
        cur_action = action_dir[step.action]
        next_state = ow.idx2pos(step.next_state)
        path += f's={cur_state}, a={cur_action}, r={step.reward}, s\'={next_state}'
        if k == len(traj)-1:
            path += ''
        else:
            path += ' -> \n'

    return path

def generate_demonstrations(ow, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=None):
    """gatheres expert demonstrations

    inputs:
    ow          Objectworld - the environment
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
            start_pos = [np.random.randint(0, ow.height), np.random.randint(0, ow.width)]

        episode = []
        ow.reset(start_pos)
        cur_state = start_pos
        cur_state, action, next_state, reward, is_done = ow.step(int(policy[ow.pos2idx(cur_state)]))
        episode.append(Step(cur_state=ow.pos2idx(cur_state), action=action, next_state=ow.pos2idx(next_state), reward=reward, done=is_done))
        # while not is_done:
        for _ in range(len_traj-1):
            cur_state, action, next_state, reward, is_done = ow.step(int(policy[ow.pos2idx(next_state)]))
            episode.append(Step(cur_state=ow.pos2idx(cur_state), action=action, next_state=ow.pos2idx(next_state), reward=reward, done=is_done))
            if is_done:
                break
        trajs.append(episode)
    return trajs

def init_object_world(args):
    # init the objectworld
    # args should contain : grid_size, n_objects, n_colours, act_random
    # return : Object world, transition matrix, ground truth value map, ground truth policy, feature_map
    print('[INFO] Initialize Object World')

    ow = Objectworld(args.grid_size, args.n_objects, args.n_colours, 1 - args.act_random)
    rewards_gt = np.reshape(ow.reward_update(), args.grid_size*args.grid_size, order='F')
    feature_map = ow.feature_matrix()

    print('[INFO] Getting ground truth values and policy via value iteration')
    values_gt, policy_gt = value_iteration(ow.P_a, rewards_gt, args.gamma, error=args.error, deterministic=True)

    return ow, ow.P_a, rewards_gt, values_gt, policy_gt, feature_map
