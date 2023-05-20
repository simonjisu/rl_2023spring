import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_infos(history, active=False, search_idx=None):
    values_gt = history[0]['values_gt']
    idxs = list(history.keys())[1:-1]
    all_values_diff = {'mean': [], 'std': []}
    for i in idxs:
        vd_mean = np.abs(history[i]['values'] - values_gt).mean()
        vd_std = np.abs(history[i]['values'] - values_gt).std()
        all_values_diff['mean'].append(vd_mean)
        all_values_diff['std'].append(vd_std)
    
    if search_idx is None:
        search_idx = idxs[-1]
    else:
        if search_idx not in idxs:
            raise KeyError(f'last_idx should be in available idxs: {idxs}')

    info_dict = {
        'rewards_gt': history[0]['rewards_gt'],
        'values_gt': history[0]['values_gt'],
        'policy_gt': history[0]['policy_gt'],
        'rewards': history[search_idx]['rewards'],
        'policy': history[search_idx]['policy'],
        'trajs': history[search_idx]['trajs'],
        'values': history[search_idx]['values'],
    }
    if active:
        info_dict['rewards_new_T'] = history[search_idx]['rewards_new_T']
        info_dict['values_new'] = history[search_idx]['values_new']
        info_dict['policy_new'] = history[search_idx]['policy_new']

    return idxs, all_values_diff, info_dict

def reshaper(args, data):
    return np.reshape(data, (args.height, args.width), order='F')

def draw_maps(args, info_dict, active=False, search_idx=None, file_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=110)
    titles = {
        'rewards_gt': 'Rewards Map (Ground Truth)',
        'values_gt': 'Value Map (Ground Truth)',
        'rewards': 'Rewards Map (Recovered)',
        'values': 'Value Map (Recovered)',
    }
    for (key, title), ax in zip(titles.items(), axes.flatten()):
        ax.set_title(title)
        sns.heatmap(reshaper(args, info_dict[key]), annot=True, fmt='.2f', ax=ax)

    suptitle = 'Active Sampling' if active else 'Random Sampling'
    if search_idx is not None:
        suptitle += f' (N_trajs={search_idx})'
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if file_path is not None:
        fig.savefig((Path(file_path) / f'maps_{"active" if active else "random"}.png'))
    plt.show()

def draw_policy(args, policy_mat, ax, H, W):
    dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '*'}
    argmax_policy = np.argmax(policy_mat, axis=1)
    pi_dirs = np.array(list(map(dirs.get, argmax_policy)))
    sns.heatmap(reshaper(args, argmax_policy), 
                annot=False, fmt='.2f', linewidths=1.0, cbar=False, cmap=ListedColormap(['white']), ax=ax)
    for i in range(H):
        for j in range(W):
            text = ax.text(
                j+0.5, i+0.5, reshaper(args, pi_dirs).reshape([H, W], order='F')[i, j],
                ha="center", va="center", color="k", fontsize=20)

def draw_acq_maps(args, info_dict, search_idx=None, file_path=None):
    H, W = args.height, args.width
    

    titles = {
        'rewards_gt': 'Rewards Map (Ground Truth)',
        'rewards': 'Rewards Map (Recovered)',
        'policy': 'Policy Map (Recovered)',
        'rewards_new_T': 'Rewards Map (Acquisition)',
        'values_new': 'Value Map (Acquisition)',
        'policy_new': 'Policy Map (Acquisition)'
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # axes_filtered = [ax for i, ax in enumerate(axes.flatten()) if i != 2]
    # axes[0, 2].axis('off')
    for (key, title), ax in zip(titles.items(), axes.flatten()):
        ax.set_title(title)
        if key in ['policy_new', 'policy']:
            draw_policy(args, info_dict[key], ax, H, W)
        else:
            sns.heatmap(reshaper(args, info_dict[key]), annot=True, fmt='.2f', ax=ax)
    suptitle = 'Rewards Map Comparison '
    if search_idx is not None:
        suptitle += f' (N_trajs={search_idx}+1)'
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if file_path is not None:
        fig.savefig((Path(file_path) / f'maps_acq_rewards.png'))
    plt.show()

def draw_evd(idxs_act, idxs_rand, vd_act, vd_rand, search_idx=None, file_path=None):
    title = 'Expected value difference'
    if search_idx is not None:
        title += f' (N_trajs={search_idx})'
    if search_idx is None:
        search_idx = len(idxs_act)
    vd_act_mean = np.array(vd_act['mean'])[:search_idx]
    vd_act_std = np.array(vd_act['std'])[:search_idx]
    vd_rand_mean = np.array(vd_rand['mean'])[:search_idx]
    vd_rand_std = np.array(vd_rand['std'])[:search_idx]
    idxs_act = idxs_act[:search_idx]
    idxs_rand = idxs_rand[:search_idx]

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(x=idxs_act, y=vd_act_mean, label='active')
    ax.fill_between(idxs_act, vd_act_mean - vd_act_std, vd_act_mean + vd_act_std, alpha=0.3)

    sns.lineplot(x=idxs_rand, y=vd_rand_mean, label='random')
    ax.fill_between(idxs_rand, vd_rand_mean - vd_rand_std, vd_rand_mean + vd_rand_std, alpha=0.3)
    
    ax.set_xlabel('Number of acquistions samples')
    ax.set_ylabel('Expected value difference')
    
    ax.set_title(title)
    # ax.set_xticks(idxs_act)
    ax.legend()
    if file_path is not None:
        fig.savefig((Path(file_path) / f'evd.png'))
    plt.show()