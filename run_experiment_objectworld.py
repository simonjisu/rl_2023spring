import numpy as np

from src.GridWorldMDP.objectworld_utils import draw_path, generate_demonstrations, init_object_world, visitation_frequency
from src.deepmaxent_irl_objectworld import run_deepmaxent_irl
from src.maxent_irl_objectworld import run_maxent_irl
from src.img_utils import Visualizer, get_evd
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
--height 16
--width 16
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate 0.05
--n_iters 20
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay 3.0
--hiddens 8 8
--device cuda
--active
--seed none
--verbose 1
"""

DEEP_MAXENT_RANDOM_ARGS = """
--exp_name deepmaxent_random
--type object
--n_objects 50
--n_colours 6
--height 16
--width 16
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate 0.05
--n_iters 20
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay 3.0
--hiddens 8 8
--device cuda
--seed none
--verbose 1
"""

# easy
DEEP_MAXENT_ACTIVE_ARGS = """
--exp_name deepmaxent_active
--type object
--n_objects 8
--n_colours 2
--height 6
--width 6
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate 0.05
--n_iters 20
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay 0.5
--hiddens 3 3
--device cuda
--active
--seed none
--verbose 1
"""

DEEP_MAXENT_RANDOM_ARGS = """
--exp_name deepmaxent_random
--type object
--n_objects 8
--n_colours 2
--height 6
--width 6
--gamma 0.9
--act_random 0.3
--n_trajs 10
--l_traj 8
--learning_rate 0.05
--n_iters 20
--alpha 0.1
--n_query 1
--r_max 1
--error 0.01
--grad_clip 0.5
--weight_decay 0.5
--hiddens 3 3
--device cuda
--seed none
--verbose 1
"""

if __name__ == '__main__':
    ARG_STRS = [DEEP_MAXENT_ACTIVE_ARGS, DEEP_MAXENT_RANDOM_ARGS]
    evd_acts = []
    evd_rands = []
    n_exp = 30
    exp_results = defaultdict(list)
    save_path = Path('exp_results')
    if not save_path.exists():
        save_path.mkdir()

    for arg_str in ARG_STRS:
        args = parse_args_str(PARSER, arg_str)
        global_progress_bar = tqdm(desc=f'[EXP] {args.exp_name}')
        exp_path = save_path / args.exp_name
        if not exp_path.exists():
            exp_path.mkdir()
        for e_num in range(n_exp):
            init_start_pos = np.random.randint(0, args.height, size=(args.n_query, 2)).tolist()
            history = run_deepmaxent_irl(args, init_start_pos=init_start_pos)
            with open(exp_path / f'{e_num}.pkl', 'wb') as f:
                pickle.dump(history, f)
            evd = get_evd(history)
            exp_results[args.exp_name].append(evd)
            global_progress_bar.update(1)

    with open('exp_results_objectworld.pkl', 'wb') as f:
        pickle.dump(exp_results, f)