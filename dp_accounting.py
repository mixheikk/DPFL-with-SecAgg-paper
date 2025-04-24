from jsonargparse import ArgumentParser
import numpy as np
from scipy.optimize import bisect
from scipy.special import binom

from utils import set_seeds
from skellam_utils import SkellamMechanismPyTorch as SkellamMechanism

# Much of the Skellam accounting code is based on the code from https://github.com/facebookresearch/dp_compression

def amplified_RDP(eps_rdp_list:list, alpha:int, q:float):
    """
    RDP amplification bound for Poisson sampling and add/remove neighbourhood, from https://arxiv.org/abs/2210.00597
    Args:
        eps_rdp_list: list of RDP epsilons corresponding to alpha=2,3,...,max_alpha
        alpha: integer >= 2
        q: Poisson sampling probability
    """
    assert alpha >= 2, f'alpha must be >= 2, got {alpha}'
    assert len(eps_rdp_list) >= alpha-1, f'eps_rdp_list must have length at least alpha-1, got {len(eps_rdp_list)} with alpha={alpha}'
    term0 = (1-q)**(alpha-1) * (1+(alpha-1)*q)
    term1 = sum([ binom(alpha, j)* (1-q)**(alpha-j) * q**j * np.exp( (j-1)*eps_rdp_list[j-2] )  for j in range(2,alpha+1) ])
    return 1/(alpha-1)*np.log((term0+term1))

def from_RDP_to_DP(eps_rdp, alpha, target_delta):
    # original Mironov 2017 conversion: Prop3 in https://arxiv.org/abs/1702.07476
    return eps_rdp-np.log(target_delta)/(alpha-1)

def get_skellam_noise_multiplier(quantization:int, num_params:int, num_clients:int, alphas:list, sampling_frac:float, n_comps:int, target_eps:float, target_delta:float):
    C = 1.0
    # do binary search to find scale
    print('Running binary search for Skellam noise scale...')
    prop_scale = bisect(f=skellam_search_fun, a=.5, b=50., args=(alphas, quantization, num_params, num_clients, sampling_frac, n_comps, target_delta, target_eps, C))
    mu = (C*prop_scale)**2
    skellam = SkellamMechanism(quantization, d=num_params, norm_bound=C, mu=mu, device='cpu', num_clients=num_clients)
    print(f'Found noise scale={prop_scale}, which gives delta={get_skellam_adp(skellam, alphas, sampling_frac, n_comps, target_delta)}')
    return prop_scale
    
def skellam_search_fun(scale, alphas, quantization, num_params, num_clients, sampling_frac, n_comps, target_delta, target_eps, C=1.0):
    mu = (C*scale)**2
    skellam = SkellamMechanism(quantization, d=num_params, norm_bound=C, mu=mu, device='cpu', num_clients=num_clients)
    min_eps = get_skellam_adp(skellam, alphas, sampling_frac, n_comps, target_delta)
    return min_eps - target_eps

def get_skellam_adp(skellam, alphas, sampling_frac, n_comps, target_delta):
    amplified_RDP_eps, i_alpha_ = get_skellam_rdp(skellam, alphas, sampling_frac, n_comps)
    min_eps = np.inf
    for i_alpha, max_alpha_ in enumerate(alphas[:i_alpha_+1]):
        total_eps = from_RDP_to_DP(amplified_RDP_eps[i_alpha], max_alpha_, target_delta)
        min_eps = min(min_eps, total_eps)
    return min_eps

def get_skellam_rdp(skellam, alphas, sampling_frac, n_comps):
    RDP_eps = skellam.renyi_div(alphas)
    amplified_RDP_eps = np.zeros(len(alphas))
    for i_alpha, max_alpha_ in enumerate(alphas):
        tmp = amplified_RDP(RDP_eps, max_alpha_, sampling_frac)
        amplified_RDP_eps[i_alpha] = n_comps * tmp
        if not np.isfinite(tmp):
            break
    return amplified_RDP_eps, i_alpha

def run_accounting(args):
    set_seeds(args.init_seed)
    print('Accounting for', args.n_comps , 'steps using Skellam mechanism with subsampling ratio', args.sampling_frac)
    alphas = np.array(list(range(2, args.max_alpha)))
    skellam_noise_sigma_ = get_skellam_noise_multiplier(quantization=args.quantization, num_params=args.num_params, num_clients=args.num_clients, alphas=alphas, sampling_frac=args.sampling_frac, n_comps=args.n_comps, target_eps=args.target_eps, target_delta=args.target_delta)
    print(f'Per-client noise value:\n{skellam_noise_sigma_/np.sqrt(args.num_clients)}')


if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")
    parser.add_argument('--init_seed', default=2303, type=int, help='Random seed.')
    parser.add_argument('--n_comps', default=10, type=int, help='Number of compositions.')
    parser.add_argument('--sampling_frac', default=0.1, type=float, help='Sampling fraction for privacy amplification.')
    parser.add_argument('--target_eps', default=1.0, type=float, help='Target epsilon.')
    parser.add_argument('--target_delta', default=1e-5, type=float, help='Target delta.')
    parser.add_argument('--num_params', default=10250, type=int, help='Number of parameters in the model.')
    parser.add_argument('--num_clients', default=10, type=int, help='Number of clients.')
    parser.add_argument('--quantization', default=32, type=int, help='Quantization level.')
    parser.add_argument('--max_alpha', default=256, type=int, help='Max alpha for RDP accounting. Try increasing this if the accounting fails or is unstable.')
    args = parser.parse_args()
    run_accounting(args)