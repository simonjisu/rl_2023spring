import numpy as np
from .gridworld import GridWorld
import torch
from .value_iteration import value_iteration

def get_new_policy(gw, P_a, policy):
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    policy_new = policy.copy()
    for s in range(N_STATES):
        s_coor = gw.idx2pos(s)
        acts = gw.get_actions(s_coor)
        for a in range(N_ACTIONS):
            if a not in acts:
                temp = policy_new[s, a]
                policy_new[s, a] = 1e-10
                policy_new[s, 4] += temp
    return policy_new

def finite_policy_evaluation(P_a, policy, reward, gamma):
    """
    inputs:
    P_a: N_STATES, N_STATES, N_ACTIONS - P_a[s_from, s_to, a]
    policy: N_STATES, N_ACTIONS
    reward: T, N_STATES
    return:
    v_t: N_STATES - v_0(s)
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    T = reward.shape[0]
    v_t = reward[T-1]
    for t in range(T-2, -1, -1):
        v_t_1 = v_t.copy()
        for s in range(N_STATES):
            v_t[s] = reward[t][s] + gamma * np.sum([np.dot(policy[s, :], P_a[s, s_to,:])*v_t_1[s_to] for s_to in range(N_STATES)])
    return v_t

def BALD_acquisition_function(model, P_a, gw, n_sample, feat_map, args):
    model.train()
    device = torch.device(args.device)
    inputs = torch.from_numpy(feat_map).float().to(device)
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    policy_D = np.zeros((n_sample, N_STATES, N_ACTIONS))
    for i in range(n_sample):
        rewards = model(inputs)
        rewards_numpy = rewards.view(-1).detach().cpu().numpy()
        _, policy_R = value_iteration(P_a, rewards_numpy, args.gamma, args.alpha, args.error, deterministic=False)
        policy_new = get_new_policy(gw, P_a, policy_R)
        rewards_new = (policy_new * -np.log(policy_new)).sum(1)
        policy_D[i] = policy_new
    policy_D = policy_D.mean(axis = 0)
    values_new = finite_policy_evaluation(P_a, policy_new, np.resize(rewards_new, (args.l_traj, len(rewards_new))), args.gamma)



    # model -> R
    # R -> policy
    # uncertainty_acquisition_function(policy)
def uncertainty_acquisition_function(
        P_a: np.ndarray, policy: np.ndarray, gw: GridWorld, gamma: float, l_traj: int
    ):
    """
    inputs:
        P_a: NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                    landing at state s1 when taking action 
                                    a at state s0
        policy: NxN_ACTIONS matrix - policy[s, a] is the prob of taking action a at state s
        gw: GridWorld - the environment
        gamma: float - RL discount factor
        l_traj: int - length of trajectory

    returns:
        rewards_new_T: Nx1 vector - reward($-\log(\pi(a_t | s_t))$) values
        values_new: Nx1 vector - state values under policy V(s_{t=0})
        policy_new: NxN_ACTIONS matrix - new policy

    adapt the policy probability in the grid world. 
    change the probability of actions that are not available in a state to 0 
    and redistribute the probability to the `stay` action.
    """

    # run value iteration
    # set thre reward to be the negative log of the policy  
    policy_new = get_new_policy(gw, P_a, policy)         
    rewards_new = (policy_new * -np.log(policy_new)).sum(1)
    values_new = finite_policy_evaluation(P_a, policy_new, np.resize(rewards_new, (l_traj, len(rewards_new))), gamma)
    return rewards_new, values_new, policy_new


def policy_evaluation(P_a, rewards, policy, gamma, error=0.001):
    """
    inputs:
        P_a: NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                    landing at state s1 when taking action
                                    a at state s0
        rewards: Nx1 vector - reward values
        policy: NxN_ACTIONS matrix - policy[s, a] is the prob of taking action a at state s
        gamma: float - RL discount factor
        error: float - threshold for a stop

    returns:
        values: Nx1 vector - estimated values
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    values = np.zeros(N_STATES)

    while True:
        values_tmp = values.copy()

        for s in range(N_STATES):
            new_value = 0.0
            for a in range(N_ACTIONS):
                transition_probs = P_a[s, :, a]
                values_next = np.sum(values * transition_probs)
                new_value += policy[s, a] * (rewards[s] + gamma * values_next)

            values[s] = new_value

        if np.abs(values_tmp - values).max() < error:
            break
    
    return values