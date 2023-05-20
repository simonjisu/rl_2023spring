import numpy as np
from .gridworld import GridWorld

def finite_policy_iteration(
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
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    L_TRAJ = l_traj
    policy_new = policy.copy()
    for s in range(N_STATES):
        for s in range(N_STATES):
            s_coor = gw.idx2pos(s)
            acts = gw.get_actions(s_coor)
            for a in range(N_ACTIONS):
                if a not in acts:
                    temp = policy_new[s, a]
                    policy_new[s, a] = 1e-10
                    policy_new[s, 4] += temp
    # run value iteration
    # set thre reward to be the negative log of the policy           
    rewards_new = -np.log(policy_new)
    rewards_new_T = (policy_new * rewards_new).sum(1)
    values = np.zeros([N_STATES, L_TRAJ+1])
    values[:, -1] = rewards_new_T

    for t in range(L_TRAJ-1, -1, -1):
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                values[s, t] += policy_new[s, a] * (rewards_new[s, a] + gamma * P_a[s, :, a].dot(values[:, t+1]))
    values_new = values[:, 0]
    return rewards_new_T, values_new, policy_new


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