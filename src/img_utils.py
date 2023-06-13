import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import math



def get_evd(history):
    values_gt = history[0]['values_gt']
    idxs = list(history.keys())[1:-1]
    evds = []
    for i in idxs:
        vd_mean = np.abs(values_gt - history[i]['values']).mean()
        evds.append(vd_mean)
    return np.array(evds)

class Visualizer:
    
    def __init__(self, history, file_path = None):
        self.history = history
        self.file_path = file_path
        self.args = history[0]['args']
        self.active = self.args.active
        self.max_val, self.min_val = math.ceil(np.max(history[0]['values_gt'])), math.floor(np.min(history[0]['values_gt']))
        self.max_r, self.min_r = math.ceil(np.max(history[0]['rewards_gt'])), math.floor(np.min(history[0]['rewards_gt']))
    
    def reshaper(self, data):
        return np.reshape(data, (self.args.height, self.args.width), order='F')
    
    def get_infos(self, search_idx):
        info_dict = {
            'rewards_gt': self.history[0]['rewards_gt'],
            'values_gt': self.history[0]['values_gt'],
            'policy_gt': self.history[0]['policy_gt'],
            'args': self.history[0]['args'],
            'rewards': self.history[search_idx]['rewards'],
            'policy': self.history[search_idx]['policy'],
            'trajs': self.history[search_idx]['trajs'],
            'values': self.history[search_idx]['values'],
        }
        if self.active:
            info_dict['rewards_new_T'] = self.history[search_idx]['rewards_new_T']
            info_dict['values_new'] = self.history[search_idx]['values_new']
            info_dict['policy_new'] = self.history[search_idx]['policy_new']

        return info_dict
    
    def draw_value_maps(self, search_idx):
        info_dict = self.get_infos(search_idx)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=110)
        titles = {
            'rewards_gt': 'Rewards Map (Ground Truth)',
            'values_gt': 'Value Map (Ground Truth)',
            'rewards': 'Rewards Map (Recovered)',
            'values': 'Value Map (Recovered)',
        }
        for (key, title), ax in zip(titles.items(), axes.flatten()):
            ax.set_title(title)
            if 'Value' in title:
                vmax, vmin = self.max_val, self.min_val
            else:
                vmax, vmin = self.max_r, self.min_r
            sns.heatmap(self.reshaper(info_dict[key]), vmax = vmax, vmin = vmin, annot=True, fmt='.2f', ax=ax)
            
        suptitle = 'Active Sampling' if self.active else 'Random Sampling'
        suptitle += f' (N_trajs={search_idx})'
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        if self.file_path is not None:
            fig.savefig((Path(self.file_path) / f'value_maps_{"active" if self.active else "random"}.png'))
        plt.show()
        return None

    def draw_acq_maps(self, search_idx):
        info_dict = self.get_infos(search_idx)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=110)
        titles = {
            'rewards_gt': 'Rewards Map (Ground Truth)',
            'rewards': 'Rewards Map (Recovered)',
            'rewards_new_T': 'Policy Entropy Map',
            'values_new': 'Acquisition function Map',
        }
        for (key, title), ax in zip(titles.items(), axes.flatten()):
            if 'Reward' in title:
                vmax, vmin = self.max_r, self.min_r
            else:
                vmax, vmin = None, None
            ax.set_title(title)
            sns.heatmap(self.reshaper(info_dict[key]), vmax = vmax, vmin = vmin, annot=True, fmt='.2f', ax=ax)

        suptitle = 'Acquisition Map'
        suptitle += f' (N_trajs={search_idx})'
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        if self.file_path is not None:
            fig.savefig((Path(self.file_path) / f'acq_maps.png'))
        plt.show()

    def draw_policy_maps(self, search_idx):
        info_dict = self.get_infos(search_idx)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=110)
        titles = {
            'values_gt': 'Value Map (Ground Truth)',
            'policy_gt': 'Policy Map (Ground Truth)',
            'values': 'Value Map (Recovered)',
            'policy': 'Policy Map (Recovered)',
        }
        dirs = {0: '>', 1: '<', 2: 'v', 3: '^', 4: '*'}
        for (key, title), ax in zip(titles.items(), axes.flatten()):
            ax.set_title(title)
            if key in ['policy_gt', 'policy']:
                if key == 'policy_gt':
                    argmax_policy = info_dict[key]
                    max_policy = np.ones(np.shape(argmax_policy))
                else:
                    argmax_policy = np.argmax(info_dict[key], axis=1)
                    max_policy = np.max(info_dict[key], axis=1)
                pi_dirs = np.array(list(map(dirs.get, argmax_policy)))
                sns.heatmap(self.reshaper(max_policy), vmin = 0, vmax = 1,
                annot=False, ax=ax)
                for i in range(self.args.height):
                    for j in range(self.args.width):
                        text = ax.text(
                            j+0.5, i+0.5, self.reshaper(pi_dirs)[i, j],
                            ha="center", va="center", color="k", fontsize=20)
            else:
                sns.heatmap(self.reshaper(info_dict[key]), vmin = self.min_val, vmax = self.max_val,
                            annot=True, fmt='.2f', ax=ax)
        suptitle = 'Active Sampling' if self.active else 'Random Sampling'
        suptitle += f' (N_trajs={search_idx})'
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        if self.file_path is not None:
            fig.savefig((Path(self.file_path) / f'policy_maps.png'))
        plt.show()
        return None

def draw_evd(evd_act, evd_rand,file_path=None):
    '''evd_act.shape = evd_rand.shape = (# of experiments, n_trajs)'''
    title = 'Expected value difference'
    n_trajs = len(evd_act[0])
    idxs = np.arange(1, n_trajs+1, 1)
    evd_act_mean = evd_act.mean(axis = 0)
    evd_act_std = evd_act.std(axis = 0)
    evd_rand_mean = evd_rand.mean(axis = 0)
    evd_rand_std = evd_rand.std(axis = 0)


    fig, ax = plt.subplots(figsize=(10, 4))

    sns.lineplot(x=idxs, y=evd_act_mean, label='active')
    ax.fill_between(idxs, evd_act_mean - evd_act_std, evd_act_mean + evd_act_std, alpha=0.3)

    sns.lineplot(x=idxs, y=evd_rand_mean, label='random')
    ax.fill_between(idxs, evd_rand_mean - evd_rand_std, evd_rand_mean + evd_rand_std, alpha=0.3)
    
    ax.set_xlabel('Number of acquistions samples')
    ax.set_ylabel('Expected value difference')
    
    ax.set_title(title)
    ax.set_xticks(idxs)
    ax.legend()
    if file_path is not None:
        fig.savefig((Path(file_path) / f'evd.png'))
    plt.show()
    
def draw_acq_maps_w_trajs(args, info_dict, traj, num_trajs=None, file_path=None):
    H, W = args.height, args.width
    scale = 0.001
    arrows = {0:(1,0), 1:(-1,0), 3:(0,1),2:(0,-1), 4:(0,0)}
    colors = ['w', 'g', 'y', 'r']

    titles = {
        'values': 'Value Map (Recovered)',
        'values_new': 'Value Map (Acquisition)',
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    gw = traj[0]['gw']

    traj_dict = {}
    for e in range(num_trajs):
        traj_dict[e] = []
        for step in traj[0]['trajs'][e]:
            cur_state = gw.idx2pos(step.cur_state)
            traj_dict[e].append((cur_state, step.action))

    for (key, title), ax in zip(titles.items(), axes.flatten()):
        ax.set_title(title)
        if key == 'values':
            sns.heatmap(reshaper(args, info_dict[key]), annot=True, fmt = '.2f', ax = ax)
            for t in ax.texts:
                trans = t.get_transform()
                offs = matplotlib.transforms.ScaledTranslation( 0, -0.25,
                                matplotlib.transforms.IdentityTransform())
                t.set_transform( offs + trans )
            for e in range(num_trajs):
                for step in traj_dict[e]:
                    print(e, step)
                    if step[1] == 4:
                        ax.plot(step[0][0]+0.5, step[0][1]+0.65, colors[e]+'o')
                    else:
                        ax.arrow(step[0][0]+0.5, step[0][1]+0.65, scale*arrows[step[1]][0], scale*arrows[step[1]][1], head_width=0.1, color = colors[e])
                    # break
            x, y = gw.idx2pos(np.argmax(reshaper(args, traj[num_trajs]['values_new'])))
            ax.plot(x+0.5, y+0.5, 'bo')
            print(gw.idx2pos(np.argmax(reshaper(args, traj[num_trajs]['values_new']))))
        else:
            sns.heatmap(reshaper(args, traj[num_trajs]['values_new']), annot=True, fmt='.2f', ax=ax)
    suptitle = 'Algorithm'
    # if num_trajs is not None:
    #     suptitle += f' (N_trajs={num_trajs}+1)'
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if file_path is not None:
        fig.savefig((Path(file_path) / f'algorithm.png'))
    plt.show()
    
