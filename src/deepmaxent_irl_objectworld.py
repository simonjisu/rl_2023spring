import numpy as np
from collections import defaultdict
from .GridWorldMDP.objectworld_utils import generate_demonstrations, init_object_world, visitation_frequency
from .deepmaxent_irl import deepmaxent_irl
from .GridWorldMDP.policy_iteration import uncertainty_acquisition_function, policy_evaluation
from .func_utils import min_max

def run_deepmaxent_irl(args, init_start_pos=None):
    """_summary_

    Args:
        args (_type_): _description_
        coor_rates (_type_): _description_
        init_start_pos (_type_, optional): _description_. Defaults to None.
    """ 

    env, P_a, rewards_gt, values_gt, policy_gt, feat_map = init_object_world(args)
    history = defaultdict(dict)

    # initial trajectories always start from random position
    print('[INFO] Initialize trajectories')
    assert args.n_query < args.n_trajs, 'ARGS.n_query must be much more smaller than N_TRAJS'
    
    if init_start_pos is None:
        trajs = generate_demonstrations(env, policy_gt, 
                                        n_trajs=args.n_query, 
                                        len_traj=args.l_traj, 
                                        rand_start=True, 
                                        start_pos=None)
    else:
        if isinstance(init_start_pos[0], list) or isinstance(init_start_pos[0], tuple):
            # multiple start points
            trajs = []
            for sp in init_start_pos:
                t = generate_demonstrations(env, policy_gt, 
                                            n_trajs=args.n_query, 
                                            len_traj=args.l_traj, 
                                            rand_start=False, 
                                            start_pos=sp)
                trajs.extend(t)
        else:
            # type(init_start_pos[0]) == int
            trajs = generate_demonstrations(env, policy_gt, 
                                            n_trajs=args.n_query, 
                                            len_traj=args.l_traj, 
                                            rand_start=False, 
                                            start_pos=init_start_pos)
            
    history[0]['env'] = env
    history[0]['P_a'] = P_a
    history[0]['rewards_gt'] = rewards_gt
    history[0]['values_gt'] = values_gt
    history[0]['policy_gt'] = policy_gt
    history[0]['args'] = args
    current_n_trajs = args.n_query
    history[current_n_trajs]['trajs'] = trajs
    # print(f'{current_n_trajs}th trajectories.')
    # print(draw_path(trajs[0], env))
    freq = visitation_frequency(trajs, args.height*args.height)
    print('Visitation Frequency')
    print(freq.reshape(args.height, args.height, order='F'))

    while True:
        print(f'[INFO - n_trajs:{current_n_trajs}] Training Deep MaxEnt IRL')
        
        # ----- run deep maxent irl -----
        rewards, policy, l2_loss, model = deepmaxent_irl(feat_map, P_a, trajs, args)
        if args.type == 'grid':
            normalize_fn = lambda x: min_max(x, is_tanh_like=False)
        elif args.type == 'object':
            normalize_fn = lambda x: min_max(x, is_tanh_like=True)
        else:
            raise NotImplementedError('Unknown environment type: {}'.format(args.type))
        
        print(rewards.reshape(args.height, args.height, order='F').round(4))
        # --- normalize rewards ---
        rewards = normalize_fn(rewards)

        history[current_n_trajs]['rewards'] = rewards   # rewards map after IRL
        history[current_n_trajs]['policy'] = policy   # policy after IRL
        history[current_n_trajs]['l2_loss'] = l2_loss   # l2 loss after IRL
        print(f'[INFO - n_trajs:{current_n_trajs}] Policy evaluation')

        # ----- policy evaluation -----    
        values = policy_evaluation(P_a, rewards_gt, policy, args.gamma, error=args.error)
        history[current_n_trajs]['values'] = values
        
        # ----- calculate EVD -----
        evd = np.abs(values_gt - values).mean()
        print(f'-- evd = {evd:.6f} ---')

        if current_n_trajs + args.n_query > args.n_trajs: # break signal
            break

        if args.active:
            print(f'[INFO - n_trajs:{current_n_trajs}] Calculating the acqusition map')

            # ----- calculate acquisition map -----
            rewards_new_T, values_new, policy_new = uncertainty_acquisition_function(P_a, policy, env, args.gamma, args.l_traj)
            # acquistion process
            # if n_query > 1 then we need to select the n_query points
            query_idxs = np.argsort(values_new)[::-1][:args.n_query]
            start_points_new = [env.idx2pos(idx) for idx in query_idxs]

            if args.verbose != 1:
                print(f'-- Acquisition Function Map when n_trajs:{current_n_trajs}--')
                print(values_new.reshape(args.height, args.height, order='F').round(4))

            print(f'[INFO - n_trajs:{current_n_trajs}] Generating a new demonstrations from {start_points_new}')

            trajs_new = []
            for sp in start_points_new:
                t_new = generate_demonstrations(env, policy_gt, 
                                                n_trajs=1, 
                                                len_traj=args.l_traj, 
                                                rand_start=False, 
                                                start_pos=sp)
                trajs_new.extend(t_new)
            history[current_n_trajs]['rewards_new_T'] = rewards_new_T
            history[current_n_trajs]['values_new'] = values_new 
            history[current_n_trajs]['policy_new'] = policy_new
        else:
            print(f'[INFO - n_trajs:{current_n_trajs}] Generating a new demonstrations from Random Points')
            trajs_new = generate_demonstrations(env, policy_gt, 
                                                n_trajs=args.n_query, 
                                                len_traj=args.l_traj, 
                                                rand_start=True, 
                                                start_pos=None)
        trajs.extend(trajs_new)
        current_n_trajs += args.n_query
        history[current_n_trajs]['trajs'] = trajs_new
        # print(f'{current_n_trajs}th trajectories.')
        # print(draw_path(trajs_new[0], env))
        freq = visitation_frequency(trajs, args.height*args.height)
        print('Visitation Frequency')
        print(freq.reshape(args.height, args.height, order='F'))
        
    return history