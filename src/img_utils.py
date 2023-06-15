import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import math


def get_evd(history):
    values_gt = history[0]['values_gt']
    idxs = list(history.keys())[1:]
    evds = []
    for i in idxs:
        vd_mean = np.abs(values_gt - history[i]['values']).mean()
        evds.append(vd_mean)
    return np.array(evds)

class Visualizer:
    
    def __init__(self, history, file_path = None, figsize=(20, 4), dpi=110):
        self.history = history
        self.file_path = file_path
        self.figsize = figsize
        self.dpi = dpi
        self.args = history[0]['args']
        self.active = self.args.active
        self.max_val, self.min_val = math.ceil(np.max(history[0]['values_gt'])), math.floor(np.min(history[0]['values_gt']))
        self.max_r, self.min_r = math.ceil(np.max(history[0]['rewards_gt'])), math.floor(np.min(history[0]['rewards_gt']))
    
    def reshaper(self, data):
        return np.reshape(data, (self.args.height, self.args.width), order='F')
    
    def get_infos(self, search_idx):
        if search_idx == -1:
            info_dict = {
            'env': self.history[0]['env'],
            'rewards_gt': self.history[0]['rewards_gt'],
            'values_gt': self.history[0]['values_gt'],
            'policy_gt': self.history[0]['policy_gt'],
        }
        else : 
            info_dict = {
                'env': self.history[0]['env'],
                'rewards_gt': self.history[0]['rewards_gt'],
                'values_gt': self.history[0]['values_gt'],
                'policy_gt': self.history[0]['policy_gt'],
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
        fig, axes = plt.subplots(1, 4, figsize=self.figsize, dpi=self.dpi)
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
        fig, axes = plt.subplots(1, 4, figsize=self.figsize, dpi=self.dpi)
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
        fig, axes = plt.subplots(1, 4, figsize=self.figsize, dpi=self.dpi)
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
    
    def objectworld_reward_policy(self, search_idx = -1):
        info_dict = self.get_infos(search_idx)
        fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=110)
        titles = {f' (N_trajs={search_idx})'}
        
        # reward
        if search_idx == -1:
            # ground-truth reward
            sns.heatmap(self.reshaper(info_dict['rewards_gt']), vmin = self.min_r, vmax = self.max_r, annot=False, fmt='.2f', ax=axes)
        else:
            sns.heatmap(self.reshaper(info_dict['rewards']), vmin = self.min_r, vmax = self.max_r, annot=False, fmt='.2f', ax=axes)

        # policy
        scale = 0.0001
        arrows = {0:(1,0), 1:(-1,0), 3:(0,1),2:(0,-1), 4:(0,0)}
        if search_idx == -1 :
            # ground-truth policy
            argmax_policy = info_dict['policy_gt']
        else:
            argmax_policy = np.argmax(info_dict['policy'], axis=1)
        for i in range(self.args.height):
            for j in range(self.args.width):
                axes.arrow(j+0.5, i+0.5, scale*arrows[self.reshaper(argmax_policy)[i, j]][0], scale*arrows[self.reshaper(argmax_policy)[i, j]][1], head_width=0.2, head_length = 0.2, edgecolor = 'black', fc = 'w')

        # object
        colors = [plt.cm.Set1(i) for i in range(self.args.n_colours)]
        ow = info_dict['env']
        for (x,y), obj in ow.objects.items():
            axes.add_patch(plt.Circle((x+0.5, y+0.5), 0.1, facecolor= colors[obj.inner_colour], alpha=1, linewidth=3, edgecolor=colors[obj.outer_colour]))

        plt.tight_layout()
        if self.file_path is not None:
            fig.savefig((Path(self.file_path) / f'objectworld_map.png'))
        plt.show()
        return None

    def draw_acq_maps_w_trajs(self, search_idx = -1, file_path=None):
        fig, axes = plt.subplots(1, 2, figsize = (10,4))        
        info_dict = self.get_infos(search_idx)
        env = info_dict['env']

        scale = 0.001
        arrows = {0:(1,0), 1:(-1,0), 3:(0,1),2:(0,-1), 4:(0,0)}
        colors = ['w', 'g', 'y', 'r']

        titles = {
            'values': 'Value Map (Recovered)',
            'values_new': 'Value Map (Acquisition)',
        }

        traj_dict = {}
        for e in range(1, search_idx):
            traj_dict[e] = []
            for step in self.history[e]['trajs'][self.args.n_query-1]:
                # print(step)
                cur_state = env.idx2pos(step.cur_state)
                traj_dict[e].append((cur_state, step.action))

        for (key, title), ax in zip(titles.items(), axes.flatten()):
            ax.set_title(title)
            if key == 'values':
                sns.heatmap(self.reshaper(info_dict[key]), annot=True, fmt = '.2f', ax = ax)
                for t in ax.texts:
                    trans = t.get_transform()
                    offs = matplotlib.transforms.ScaledTranslation( 0, -0.25,
                                    matplotlib.transforms.IdentityTransform())
                    t.set_transform( offs + trans )
                for e in range(1, search_idx):
                    for step in traj_dict[e]:
                        # print(e, step)
                        if step[1] == 4:
                            ax.plot(step[0][0]+0.5, step[0][1]+0.65, colors[e]+'o')
                        else:
                            ax.arrow(step[0][0]+0.5, step[0][1]+0.65, scale*arrows[step[1]][0], scale*arrows[step[1]][1], head_width=0.1, color = colors[e])
                        # break
                x, y = env.idx2pos(np.argmax(self.reshaper(info_dict['values_new'])))
                ax.plot(x+0.5, y+0.5, 'bo')
                # print(env.idx2pos(np.argmax(self.reshaper(info_dict['values_new']))))
            else:
                sns.heatmap(self.reshaper(info_dict['values_new']), annot=True, fmt='.2f', ax=ax)
        suptitle = 'Algorithm'
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        if file_path is not None:
            fig.savefig((Path(file_path) / f'algorithm.png'))
        plt.show()
    
    
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
    

    
