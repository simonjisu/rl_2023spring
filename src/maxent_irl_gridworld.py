import numpy as np
import argparse
from collections import defaultdict, namedtuple
from .mdp.gridworld import GridWorld 
from .mdp.value_iteration import value_iteration
from .mdp.policy_iteration import finite_policy_iteration, policy_evaluation
from .maxent_irl import maxent_irl

Step = namedtuple('Step','cur_state action next_state reward done')

def draw_path(traj, gw):
    # Step = namedtuple('Step','cur_state action next_state reward done')
    path = ''
    action_dir = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
    for step in traj:
        cur_state = gw.idx2pos(step.cur_state)
        cur_action = action_dir[step.action]
        next_state = gw.idx2pos(step.next_state)
        path += f's={cur_state}, a={cur_action}, r={step.reward}, s\'={next_state}->'

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


def run_maxent_irl(ARGS, init_start_pose=None):
  # rewards_gt = np.reshape(rmap_gt, H*W, order='F')
  n_states = ARGS.height * ARGS.width
  gw, P_a, rewards_gt, values_gt, policy_gt = init_grid_world(ARGS)
  history = defaultdict(dict)
  # use identity matrix as feature
  feat_map = np.eye(n_states)
  # initial trajectories always start from random position
  print('[INFO] Initialize trajectories')
  assert ARGS.n_query < ARGS.n_trajs, 'ARGS.n_query must be much more smaller than N_TRAJS'
  if init_start_pose is None:
    trajs = generate_demonstrations(gw, policy_gt, 
                                    n_trajs=ARGS.n_query, len_traj=ARGS.l_traj, rand_start=True, start_pos=None)
  else:
    if isinstance(init_start_pose[0], list) or isinstance(init_start_pose[0], tuple):
      # multiple start points
      trajs = []
      for sp in init_start_pose:
        t = generate_demonstrations(gw, policy_gt, 
                                    n_trajs=ARGS.n_query, 
                                    len_traj=ARGS.l_traj, 
                                    rand_start=False, 
                                    start_pos=sp)
        trajs.extend(t)
    else:
      # type(init_start_pose[0]) == int
      trajs = generate_demonstrations(gw, policy_gt, 
                                      n_trajs=ARGS.n_query, 
                                      len_traj=ARGS.l_traj, 
                                      rand_start=False, 
                                      start_pos=init_start_pose)
  history[0]['gw'] = gw
  history[0]['P_a'] = P_a
  history[0]['rewards_gt'] = rewards_gt
  history[0]['values_gt'] = values_gt
  history[0]['policy_gt'] = policy_gt
  history[0]['args'] = ARGS
  current_n_trajs = ARGS.n_query
  history[current_n_trajs]['trajs'] = trajs
  print(f'{current_n_trajs}th trajectories.')
  print(draw_path(trajs[0], gw))
  while True:
    print(f'[INFO - n_trajs:{current_n_trajs}] Training MaxEnt IRL')
    rewards, policy = maxent_irl(feat_map, P_a, ARGS.gamma, trajs, lr=ARGS.learning_rate, c = ARGS.grad_clip, lam = ARGS.lam, n_iters=ARGS.n_iters, alpha=ARGS.alpha, error=ARGS.error)
    print(f'--Reward Map (Recovered) when n_trajs:{current_n_trajs}--')
    print(rewards.reshape(ARGS.height, ARGS.width, order='F'))
    history[current_n_trajs]['rewards'] = rewards   # rewards map after IRL
    history[current_n_trajs]['policy'] = policy   # policy after IRL

    print(f'[INFO - n_trajs:{current_n_trajs}] Policy evaluation')
    values = policy_evaluation(P_a, rewards_gt, policy, ARGS.gamma, error=ARGS.error)
    history[current_n_trajs]['values'] = values

    if current_n_trajs + ARGS.n_query > ARGS.n_trajs: # break signal
      break

    if ARGS.active:
      print(f'[INFO - n_trajs:{current_n_trajs}] Calculating the acqusition map')
      rewards_new_T, values_new, policy_new = finite_policy_iteration(P_a, policy, gw, ARGS.gamma, ARGS.l_traj)
  
      # acquistion process
      # if n_query > 1 then we need to select the n_query points
      query_idxs = np.argsort(values_new)[::-1][:ARGS.n_query]
      start_points_new = [gw.idx2pos(idx) for idx in query_idxs]
      print(f'-- Acquisition Function Map when n_trajs:{current_n_trajs}--')
      print(values_new.reshape(ARGS.height, ARGS.width, order='F'))
      
      print(f'[INFO - n_trajs:{current_n_trajs}] Generating a new demonstrations from {start_points_new}')
      trajs_new = []
      for sp in start_points_new:
        t_new = generate_demonstrations(gw, policy_gt, 
                                        n_trajs=1, 
                                        len_traj=ARGS.l_traj, 
                                        rand_start=False, 
                                        start_pos=sp)
        trajs_new.extend(t_new)
      history[current_n_trajs]['rewards_new_T'] = rewards_new_T
      history[current_n_trajs]['values_new'] = values_new 
      history[current_n_trajs]['policy_new'] = policy_new

    else:
      print(f'[INFO - n_trajs:{current_n_trajs}] Generating a new demonstrations from Random Points')
      trajs_new = generate_demonstrations(gw, policy_gt, 
                                          n_trajs=ARGS.n_query, len_traj=ARGS.l_traj, rand_start=True, start_pos=None)

    trajs.extend(trajs_new)
    current_n_trajs += ARGS.n_query

    history[current_n_trajs]['trajs'] = trajs_new
    print(f'{current_n_trajs}th trajectories.')
    print(draw_path(trajs_new[0], gw))

  return history


def init_grid_world(ARGS):
  # init the gridworld
  # rmap_gt is the ground truth for rewards
  print('[INFO] Initialize Grid World')
  rmap_gt = np.zeros([ARGS.height, ARGS.width])
  rmap_gt[ARGS.height-2, ARGS.width-2] = ARGS.r_max
  rmap_gt[0, ARGS.width-1] = ARGS.r_max / 2
  rmap_gt[1, 1] = ARGS.r_max / 2

  gw = GridWorld(rmap_gt, {}, 1 - ARGS.act_random)

  rewards_gt = np.reshape(rmap_gt, ARGS.height*ARGS.width, order='F')
  P_a = gw.get_transition_mat()
  print('[INFO] Getting ground truth values and policy via value teration')
  values_gt, policy_gt = value_iteration(P_a, rewards_gt, ARGS.gamma, error=ARGS.error, deterministic=True)
  
  return gw, P_a, rewards_gt, values_gt, policy_gt

if __name__ == "__main__":
  PARSER = argparse.ArgumentParser(description=None)
  PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
  PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
  PARSER.add_argument('-g', '--gamma', default=0.8, type=float, help='discount factor')
  PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
  PARSER.add_argument('-t', '--n_trajs', default=100, type=int, help='number of expert trajectories')
  PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
  PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
  PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
  PARSER.set_defaults(rand_start=False)
  PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
  PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
  PARSER.add_argument('-act', '--active', action='store_true', help='active learning setting')  # store true
  PARSER.add_argument('-al', '--alpha', default=1.0, type=float, help='temperature parameter for value iteration')
  PARSER.add_argument('-nq', '--n_query', default=1, type=int, help='number of queries to the expert(n_demonstrations)')
  PARSER.add_argument('-rm', '--r_max', default=1, type=int, help='maximum reward value')
  PARSER.add_argument('-er', '--error', default=0.01, type=float, help='error threshold for policy evaluation and value iteration')
  ARGS = PARSER.parse_args()
  print(ARGS)