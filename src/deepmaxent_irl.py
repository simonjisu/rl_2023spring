import numpy as np
import torch
import torch.nn as nn
from collections import Counter

from .GridWorldMDP.value_iteration import value_iteration
from tqdm import tqdm as tqdm_progressbar
from tqdm.notebook import tqdm as tqdm_notebook_progressbar

class DeepIRLFC(nn.Module):
    def __init__(self, input_dim, hiddens: list[int], output_dim: int=1):
        super(DeepIRLFC, self).__init__()
        self.input_dim = input_dim
        self.hiddens = hiddens
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hiddens[0]),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(self.hiddens[0], self.hiddens[1]),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(self.hiddens[1], self.output_dim),  # reward
            # nn.Tanh()
        )

    def forward(self, x):
        """Get reward"""
        x = self.layers(x)
        return x

class DeepIRLCNN(nn.Module):
    def __init__(self, input_dim, hiddens: list[int], output_dim: int=1):
        super(DeepIRLCNN, self).__init__()
        self.input_dim = input_dim
        self.hiddens = hiddens
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, hiddens[0], 5, padding=2),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Conv1d(hiddens[0], hiddens[1], 3, padding=1),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Conv1d(hiddens[1], output_dim, 1, padding=0),
            # nn.Tanh()
        )

    def forward(self, x):
        x = x.permute(1, 0)[None,]  # (1, D, S)
        x = self.layers(x)  # (1, 1, S)
        x = x.squeeze().unsqueeze(-1)  # (S, 1)
        return x

# Residual Network
def build_block(input_dim, output_dim, kernel_size=3, stride=1, padding=1, dropout=0.25):
    return [
        nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ELU(),
        nn.Dropout(dropout)
    ]

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c,  downsample=None):
        super().__init__()
        self.conv = nn.Sequential(
            *build_block(in_c, out_c, kernel_size=3, stride=1, padding=1)
        )
        self.act = nn.ELU()
        self.downsample = downsample
        self.out_c = out_c

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out


class DeepIRLResNet(nn.Module):
    def __init__(self, input_dim, hiddens: list[int], output_dim: int=1):
        super(DeepIRLCNN, self).__init__()
        self.input_dim = input_dim
        self.hiddens = hiddens
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        
        # init block always build additional blocks as the same dim with hiddens[0]
        # e.g. out_channel = hiddens = [32, 32, 16, 16, 8]
        # first layer channel: input_dim > 32
        # layers = {32: 2, 16: 2, 8: 1}
        self.in_c = hiddens[0]
        blocks_count = Counter(hiddens)
        self.layers.append(nn.Sequential(*build_block(input_dim, self.in_c)))
        for out_c, n_blocks in blocks_count.items():
            self.layers.append(self.build_layers(out_c, n_blocks))
            
        # last layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(self.in_c, output_dim, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        ))

    def build_layers(self, out_c, n_blocks):
        downsample = None
        if self.in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c, kernel_size=1, stride=1, padding=0),
                nn.ELU()
            )
        layers = []
        layers.append(ResidualBlock(self.in_c, out_c, downsample))
        self.in_c = out_c
        for l in range(1, n_blocks):
            layers.append(ResidualBlock(self.in_c, out_c, downsample=None))
        return nn.Sequential(*layers)
              
    def forward(self, x):
        """Get reward"""
        grid_size = int(np.sqrt(x.size(0)))
        # x: SxD -> 1xDxHxW
        x = x.view(grid_size, grid_size, self.input_dim).permute(2, 1, 0)[None,]
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.squeeze()  # 1x1xHxW -> HxW
        # x: 1xHxW -> Sx1
        return x.permute(1, 0).reshape(-1, 1)

def compute_state_visition_freq(P_a: np.ndarray, trajs, policy: np.ndarray, deterministic:bool=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
    p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    N = len(trajs)
    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T]) 

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu /= N
    # mu[:,0] = mu[:,0]/len(trajs)
    for t in range(T-1):        
        for s in range(N_STATES):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
    p = np.sum(mu, 1)/T
    return p

def demo_svf(trajs, n_states):
    """
    compute state visitation frequences from demonstrations

    input:
    trajs   list of list of Steps - collected from expert
    returns:
    p       Nx1 vector - state visitation frequences   
    """

    p = np.zeros(n_states)
    for traj in trajs:
        for step in traj:
            p[step.cur_state] += 1
    p = p/np.sum(p)
    return p

def get_grad_theta(args, rewards, model, grad_r):
    all_grads = torch.autograd.grad(rewards, model.parameters(), grad_outputs=-grad_r, retain_graph=True)
    l2_loss = torch.stack([torch.sum(p.pow(2))/2 for p in model.parameters()]).sum()
    l2_grad = torch.autograd.grad(l2_loss, model.parameters(), retain_graph=True)
    all_grad_l2 = [args.weight_decay*l2_grad[i]+all_grads[i] for i in range(len(all_grads))]
    global_norm = torch.sqrt(torch.stack([torch.norm(g).pow(2) for g in all_grad_l2]).sum())
    clip_coef = args.grad_clip / max(global_norm, args.grad_clip)
    grad_theta = [g * clip_coef for g in all_grad_l2]
    return grad_theta, l2_loss

def apply_gradient(model, grad_theta, args):
    for p, g in zip(model.parameters(), grad_theta):
        p.data -= args.learning_rate * g

def deepmaxent_irl(feat_map, P_a, trajs, args, model=None):    
    """
    Deep Maximum Entropy Inverse Reinforcement Learning (Deep Maxent IRL)

    inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                        landing at state s1 when taking action 
                                        a at state s0
    trajs       a list of demonstrations
    args:
        hiddens         list[int] - hidden layer sizes
        gamma           float - RL discount factor
        
        learning_rate   float - learning rate
        grad_clip       float - clip
        n_iters         int - number of optimization steps
        weight_decay    float - weight decay

    returns
        rewards     Nx1 vector - recoverred state rewards
    """
    device = torch.device(args.device)
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    mu_D = demo_svf(trajs, N_STATES)
    inputs = torch.from_numpy(feat_map).float().to(device)

    if model is None:
        if args.architecture == 'dnn':
            model = DeepIRLFC(input_dim=feat_map.shape[1], hiddens=args.hiddens, output_dim=1).to(device)
        elif args.architecture == 'cnn':
            model = DeepIRLCNN(input_dim=feat_map.shape[1], hiddens=args.hiddens, output_dim=1).to(device)
        else:
            raise NotImplementedError('Unknown architecture: {}'.format(args.architecture))
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )

    # training
    if args.verbose == 1:
        progressbar = tqdm_progressbar(range(args.n_iters), total=args.n_iters, leave=False)
    elif args.verbose == 2:
        progressbar = tqdm_notebook_progressbar(range(args.n_iters), total=args.n_iters, leave=True)
    else:
        progressbar = range(args.n_iters)

    model.train()
    for iteration in progressbar:
        # zero gradients
        optimizer.zero_grad()

        # compute reward
        rewards = model(inputs)
        rewards_numpy = rewards.view(-1).detach().cpu().numpy()
        
        # approximate value iteration
        _, policy = value_iteration(P_a, rewards_numpy, 
                                    gamma=args.gamma, 
                                    alpha=args.alpha, 
                                    error=args.error, 
                                    deterministic=False)

        # propagate policy: expected state visitation frequencies
        mu_exp = compute_state_visition_freq(P_a, trajs, policy, deterministic=False)

        # compute gradient on rewards
        grad_r = mu_D - mu_exp
        grad_r = torch.from_numpy(grad_r).float().view(-1, 1).to(device)
        rewards.backward(-grad_r)

        # for records calculate l2 loss
        l2_loss = torch.stack([torch.sum(p.pow(2))/2 for p in model.parameters()]).sum().item()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) 

        # update parameters
        optimizer.step()
        # grad_theta, l2_loss = get_grad_theta(args, rewards, model, grad_r)
        # apply_gradient(model, grad_theta, args)

        if args.verbose != 0:
            progressbar.set_description_str(f'l2 loss = {l2_loss:.4f}')
        
        if (args.verbose == 0) and (iteration % (args.n_iters/20) == 0):
            print(f'iteration: {iteration}/{args.n_iters} l2 loss = {l2_loss:.4f}')
            # print('rewards')
            # print(rewards_numpy.reshape(args.height, args.width, order='F'))
            # print(f'Grad r')
            # print(grad_r.view(-1).detach().cpu().numpy().round(6))
            # for ln, lp in model.named_parameters():
            #     print(f'gradient of layer {ln}')
            #     print(lp.grad[:10])
            # print('Grad Theta')
            # print(grad_theta[0].view(-1).detach().cpu().numpy().round(6)[:10])
    
    model.eval()
    l2_loss = torch.stack([torch.sum(p.pow(2))/2 for p in model.parameters()]).sum().detach().cpu().numpy()
    with torch.no_grad():
        rewards = model(inputs)
    rewards_numpy = rewards.view(-1).detach().cpu().numpy()

    _, policy = value_iteration(P_a, rewards_numpy, 
                                gamma=args.gamma, 
                                alpha=args.alpha, 
                                error=args.error, 
                                deterministic=False)
    # print(f'unnormed rewards')
    # print(rewards_numpy.reshape(args.height, args.width, order='F'))
    # return rewards_numpy, policy, l2_loss
    return rewards_numpy, policy, l2_loss, model