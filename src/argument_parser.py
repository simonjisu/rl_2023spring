import argparse

def get_parser():
    PARSER = argparse.ArgumentParser(description=None)
    PARSER.add_argument('-exp', '--exp_name', type=str, help='base arguments')
    PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
    PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
    PARSER.add_argument('-g', '--gamma', default=0.8, type=float, help='discount factor')
    PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
    PARSER.add_argument('-t', '--n_trajs', default=100, type=int, help='number of expert trajectories')
    PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
    PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
    PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
    PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
    PARSER.add_argument('-act', '--active', action='store_true', help='active learning setting')  # store true
    PARSER.add_argument('-al', '--alpha', default=1.0, type=float, help='temperature parameter for value iteration')
    PARSER.add_argument('-nq', '--n_query', default=1, type=int, help='number of queries to the expert(n_demonstrations)')
    PARSER.add_argument('-rm', '--r_max', default=1, type=int, help='maximum reward value')
    PARSER.add_argument('-er', '--error', default=0.01, type=float, help='error threshold for policy evaluation and value iteration')
    PARSER.add_argument('-c', '--grad_clip', default=0.5, type=float, help='Gradient Clipping maximum L1 norm')
    PARSER.add_argument('-wd', '--weight_decay', default=0.5, type=float, help='L2 Regularizing Constant')
    PARSER.add_argument('-d', '--device', default='cpu', type=str, help='device to use for tensor operations; cpu or cuda')
    PARSER.add_argument('-hs', '--hiddens', nargs='+', default=[32, 32], type=int, help='hidden layer sizes for the network')
    PARSER.add_argument('-verb', '--verbose', default=2, type=int, help='0: print, 1: tqdm, 2: tqdm_notebook')

    return PARSER

def parse_args_str(parser, args_str):
    args = parser.parse_args(args_str.split())
    return args
