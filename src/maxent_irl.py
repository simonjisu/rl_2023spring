'''
Implementation of maximum entropy inverse reinforcement learning in
  Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
  https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
  This implementation is largely influenced by Matthew Alger's maxent implementation here:
  https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

Modified from Yiren Lu (luyirenmax@gmail.com), May 2017
'''
import numpy as np
from .GridWorldMDP.value_iteration import value_iteration
from .utils import normalize
from tqdm import tqdm as tqdm_progressbar
from tqdm.notebook import tqdm as tqdm_notebook_progressbar

def gradient_clip(grad, c):
    """"Gradient clipping
    input: gradient
    returns: cliped gradient with L1 norm <= c"""
    if c == 0.0:
        return grad
    new_grad = grad.copy()
    for i in range(len(grad)):
        if grad[i] > c:
            new_grad[i] = c
        elif grad[i] < -c:
            new_grad[i] = -c
    return new_grad


def compute_state_visition_freq(P_a: np.ndarray, trajs, policy: np.ndarray, deterministic:bool=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
    p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T]) 

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:,0] = mu[:,0]/len(trajs)

    for s in range(N_STATES):
        for t in range(T-1):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
    p = np.sum(mu, 1)
    return p


def maxent_irl(feat_map, P_a, trajs, args):#, gamma, lr, c, lam, n_iters, alpha=1.0, error=0.01):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

    inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                        landing at state s1 when taking action 
                                        a at state s0
    trajs       a list of demonstrations
    args:
        gamma           float - RL discount factor
        
        learning_rate   float - learning rate
        grad_clip       float - clip
        n_iters         int - number of optimization steps
        weight_decay    float - weight decay

    returns
        rewards     Nx1 vector - recoverred state rewards
    """

    # init parameters
    theta = np.random.uniform(size=(feat_map.shape[1],))

    # calc feature expectations
    feat_exp = np.zeros([feat_map.shape[1]])
    for episode in trajs:
        for step in episode:
            feat_exp += feat_map[step.cur_state,:]
    feat_exp = feat_exp/len(trajs)

    # training
    if args.verbose == 1:
        progressbar = tqdm_progressbar(range(args.n_iters), total=args.n_iters)
    elif args.verbose == 2:
        progressbar = tqdm_notebook_progressbar(range(args.n_iters), total=args.n_iters)
    else:
        progressbar = range(args.n_iters)

    for iteration in progressbar:
        # compute reward function
        rewards = np.dot(feat_map, theta)

        # compute policy
        _, policy = value_iteration(P_a, rewards, args.gamma, alpha=args.alpha, error=args.error, deterministic=False)

        # compute state visition frequences
        svf = compute_state_visition_freq(P_a, trajs, policy, deterministic=False)

        # compute gradients
        grad = feat_exp - feat_map.T.dot(svf) - args.weight_decay * theta
        grad = gradient_clip(grad, args.grad_clip)

        # update params
        theta += args.learning_rate * grad
        if (args.verbose == 0) and (iteration % (args.n_iters/20) == 0):
            print('iteration: {}/{}'.format(iteration, args.n_iters), flush=True)
            # print('theta: \n {}'.format(theta.round(3).reshape(6, 6)))
            # print('grad: \n {}'.format(grad.round(3).reshape(6, 6)))

    rewards = np.dot(feat_map, theta)
    _, policy = value_iteration(P_a, rewards, args.gamma, alpha=args.alpha, error=args.error, deterministic=False)

    return normalize(rewards), policy


