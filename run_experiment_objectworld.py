import torch
import numpy as np

from src.deepmaxent_irl import DeepIRLFC, DeepIRLCNN
from src.deepmaxent_irl_objectworld import run_deepmaxent_irl
from src.img_utils import get_evd
from src.argument_parser import get_parser, parse_args_str

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import pickle

PARSER = get_parser()

DEEP_MAXENT_ACTIVE_ARGS = """
--exp_name deepmaxent_active
--type object
--n_objects 50
--n_colours 6
--height {grid_size}
--width {grid_size}
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate {learnging_rate}
--n_iters {n_iters}
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay {weight_decay}
--hiddens 8 8
--device cuda
--active
--seed {seed}
--verbose 1
--architecture cnn
"""

DEEP_MAXENT_RANDOM_ARGS = """
--exp_name deepmaxent_random
--type object
--n_objects 50
--n_colours 6
--height {grid_size}
--width {grid_size}
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate {learnging_rate}
--n_iters {n_iters}
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay {weight_decay}
--hiddens 8 8
--device cuda
--seed {seed}
--verbose 1
--architecture cnn
"""

DEEP_MAXENT_BALD_ARGS = """
--exp_name deepmaxent_bald
--type object
--n_objects 50
--n_colours 6
--height {grid_size}
--width {grid_size}
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate {learnging_rate}
--n_iters {n_iters}
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay {weight_decay}
--hiddens 8 8
--device cuda
--new_active
--seed {seed}
--verbose 1
--architecture cnn
"""

def create_seeds(n_exp, n_train, n_test):
    # Set Seed setting
    seeds = np.arange(0, n_exp*(n_train+n_test)*3)
    np.random.shuffle(seeds)
    exp_infos = []
    for e_num in range(n_exp):
        train_seeds = seeds[e_num*n_train:(e_num+1)*n_train]
        test_seeds = seeds[n_exp*n_train+e_num*n_test:n_exp*n_train+(e_num+1)*n_test]
        train_init_start_pos = [np.random.randint(0, 16, size=(1, 2)).tolist() for _ in range(n_train)]
        test_init_start_pos = [np.random.randint(0, 16, size=(1, 2)).tolist() for _ in range(n_test)]

        info = defaultdict(list)
        for train_seed, train_init_start in zip(train_seeds, train_init_start_pos):
            info['train'].append((train_seed, train_init_start))
        for test_seed, test_init_start in zip(test_seeds, test_init_start_pos):
            info['test'].append((test_seed, test_init_start))
        exp_infos.append(info)
    with open('exp_infos.pkl', 'wb') as f:
        pickle.dump(exp_infos, f)

def main(exp_infos, arg_str_base, exp_name, n_exp, n_train, n_test, grid_size, exp_args):
    global_progress_bar = tqdm(total=n_exp*(n_train+n_test))
    save_path = Path('exp_results')
    if not save_path.exists():
        save_path.mkdir()
    exp_results = []

    arch_dict = {'dnn': DeepIRLFC, 'cnn': DeepIRLCNN}
    
    for e_num in range(n_exp):
        exp_info = exp_infos[e_num]
        global_progress_bar.set_description_str(f'[EXP-{e_num}]')
        exp_path = save_path / f'{exp_name}_{e_num}'
        if not exp_path.exists():
            exp_path.mkdir()
            
        # Training
        res_info = defaultdict(list)
        for i, (train_seed, train_init_start) in enumerate(exp_info['train']):
            global_progress_bar.set_description_str(f'[EXP-{e_num}] train {i}-{train_seed}')
            arg_str = arg_str_base.format(grid_size=grid_size, seed=train_seed, 
                                          learnging_rate=exp_args['learning_rate'], 
                                          weight_decay=exp_args['weight_decay'],
                                          n_iters=exp_args['n_iters'])
            args = parse_args_str(PARSER, arg_str)
            model_arch = arch_dict[args.architecture]
            if i == 0:
                init_model = None
            else:
                init_model = model_arch(2*args.n_colours, args.hiddens, 1).to(torch.device(args.device))
                init_model.load_state_dict(history[args.n_trajs]['model_paramaters'])
            history = run_deepmaxent_irl(args, 
                                         init_start_pos=train_init_start, 
                                         init_model=init_model, 
                                         is_train=True)
            res_info['train_evds'].append(get_evd(history))
            with open(exp_path / f'{i}-train.pkl', 'wb') as f:
                pickle.dump(history, f)
            global_progress_bar.update(1)

        # Testing
        with open(exp_path / f'{n_train-1}-train.pkl', 'rb') as f:
            history = pickle.load(f)
        
        model_arch = arch_dict[args.architecture]
        init_model = model_arch(2*args.n_colours, args.hiddens, 1).to(torch.device(args.device))
        init_model.load_state_dict(history[args.n_trajs]['model_paramaters'])
        for i, (test_seed, test_init_start) in enumerate(exp_info['test']):
            global_progress_bar.set_description_str(f'[EXP-{e_num}] test {i}-{test_seed}')
            arg_str = arg_str_base.format(grid_size=grid_size, seed=test_seed, 
                                          learnging_rate=exp_args['learning_rate'], 
                                          weight_decay=exp_args['weight_decay'],
                                          n_iters=exp_args['n_iters'])
            args = parse_args_str(PARSER, arg_str)
            history = run_deepmaxent_irl(args, 
                                         init_start_pos=test_init_start, 
                                         init_model=init_model, 
                                         is_train=False)
            res_info['test_evds'].append(get_evd(history))
            with open(exp_path / f'{i}-test.pkl', 'wb') as f:
                pickle.dump(history, f)
            global_progress_bar.update(1)

        exp_results.append(res_info)
        
    with open(f'{exp_name}-exp_results_objectworld.pkl', 'wb') as f:
        pickle.dump(exp_results, f)

if __name__ == '__main__':
    ARG_STRS = [DEEP_MAXENT_RANDOM_ARGS, DEEP_MAXENT_ACTIVE_ARGS, DEEP_MAXENT_BALD_ARGS]
    EXP_NAMES = ['deepmaxent_random', 'deepmaxent_active', 'deepmaxent_bald']
    n_exp = 1
    n_train = 8
    n_test = 4
    grid_size = 8
    exp_args = dict(
        n_iters = 100,
        learning_rate = 0.02,
        weight_decay = 0.5
    )
    
    if not Path('exp_infos.pkl').exists():
        create_seeds(n_exp, n_train, n_test)
    with open('exp_infos.pkl', 'rb') as f:
        exp_infos = pickle.load(f)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=int, help='base arguments')
    parser.add_argument('-i', '--index', type=int, default=0, help='restart index of seed')
    args = parser.parse_args()
    main(exp_infos, ARG_STRS[args.exp], EXP_NAMES[args.exp], n_exp, n_train, n_test, grid_size, exp_args)