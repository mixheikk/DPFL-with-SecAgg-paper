"""Skellam mechanism and related util funs, code originally from https://github.com/facebookresearch/dp_compression
"""

import torch
import numpy as np
import sys
import math
import scipy.optimize as optimize
from hadamard_transform import pad_to_power_of_2, randomized_hadamard_transform, inverse_randomized_hadamard_transform

class SkellamMechanismPyTorch:
    '''
    Skellam mechanism from https://arxiv.org/pdf/2110.04995.pdf
    '''
    
    def __init__(self, budget, d, norm_bound, mu, device, num_clients=1, s=None, do_random_rotation=False):
        
        self.budget = budget
        self.d = d
        self.expanded_d = int(math.pow(2, math.ceil(math.log2(d))))
        self.norm_bound = norm_bound
        self.mu = mu
        if s is None:
            self.s = self.compute_s(num_clients)
        else:
            self.s = s
        self.inflated_bound = post_rounding_l2_norm_bound(self.expanded_d, l2_norm_bound=self.s*self.norm_bound, beta=np.exp(-.5))
        self.clip_min = -int(math.pow(2, budget - 1))
        self.clip_max = int(math.pow(2, budget - 1)) - 1
        self.device = device
        self.seed = None
        self.do_random_rotation = do_random_rotation
    
    def compute_s(self, num_clients, k=3, rho=1, DIV_EPSILON=1e-22):
        """
        Adapted from https://github.com/google-research/federated/blob/master/distributed_dp/accounting_utils.py
        This computes scaling to avoid overflows in finite field arithmetic.
        Originally from Eq.62 in https://arxiv.org/abs/2102.06387, this needs to be smaller than r^2, when r is the range of the modular clipping, so this controls the error from modular clipping
        k: signal bound multiplier from Skellam paper https://arxiv.org/abs/2110.04995, small integer
        rho: how good flattening matrix is, should be 1 or at most 2 according to DDG paper https://arxiv.org/abs/2102.06387
        """
        def mod_min(gamma):
            var = rho / self.d * (num_clients * self.norm_bound)**2
            var += (gamma**2 / 4 + self.mu) * num_clients
            return k * math.sqrt(var)

        def gamma_opt_fn(gamma):
            return (math.pow(2, self.budget) - 2 * mod_min(gamma) / (gamma + DIV_EPSILON))**2

        gamma_result = optimize.minimize_scalar(gamma_opt_fn)
        if not gamma_result.success:
            raise ValueError('Cannot compute scaling factor.')
        return 1. / gamma_result.x
    
    def renyi_div(self, alphas, l1_norm_bound=None, l2_norm_bound=None):
        """
        Computes Renyi divergence of the Skellam mechanism.
        """
        if l2_norm_bound is None:
            l2_norm_bound = self.norm_bound
        if l1_norm_bound is None:
            l1_norm_bound = self.norm_bound * min(math.sqrt(self.expanded_d), self.norm_bound)
        epsilons = np.zeros(alphas.shape)
        B1 = 3 * l1_norm_bound / (2 * self.s ** 3 * self.mu ** 2)
        B2 = 3 * l1_norm_bound / (2 * self.s * self.mu)
        for i in range(len(alphas)):
            alpha = alphas[i]
            epsilon = alpha * self.norm_bound ** 2 / (2 * self.mu)
            B3 = (2 * alpha - 1) * self.norm_bound ** 2 / (4 * self.s ** 2 * self.mu ** 2)
            epsilons[i] = epsilon + min(B1 + B3, B2)
        return epsilons
    
    def dither(self, x):
        k = torch.floor(x).to(self.device)
        prob = x - k
        max_iter = 1*10**5
        # use expanded dim in calculating post rounding bound
        for i in range(max_iter):
            output = k + (torch.rand(k.shape).to(self.device) < prob)
            if output.norm() <= self.inflated_bound: # use post rounding inflated norm bound
                break
        if i == max_iter - 1:
            print('dithering did not converge, final norm:', output.norm(), 'target norm:', self.s * self.norm_bound)
            raise RuntimeError("Dithering did not converge")
        return output.long()
    
    def privatize(self, x, same_rotation_batch=False, final_microbatch=True):
        # add some margin due to clipping rounding issues
        assert torch.all( torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=False, dtype=torch.float32) <= self.norm_bound), 'Vector norm bound exceeded, check clipping!'
        assert x.size(1) == self.d
        prng = torch.Generator(device=self.device)
        self.seed = prng.seed()
        if self.do_random_rotation:
            x = randomized_hadamard_transform(pad_to_power_of_2(x), prng.manual_seed(self.seed), same_rotation_batch)
        z = torch.zeros(x.size()).long().to(self.device)
        for i in range(x.shape[0]):
            z[i] = self.dither(self.s * x[i])
            # rescale mu to use the inflated post rounding norm bound
            if self.mu > 0 and final_microbatch and i == 0:
                scale = (self.s*self.inflated_bound/self.norm_bound)**2 * self.mu * torch.ones_like(z[i]).to(self.device)
                z[i] += (torch.poisson(scale) - torch.poisson(scale)).to(self.device).long()
        z = torch.remainder(z - self.clip_min, self.clip_max - self.clip_min) + self.clip_min
        return z
    
    def decode(self, z, same_rotation_batch=True):
        assert self.seed is not None, "Must call privatize before decode."
        prng = torch.Generator(device=self.device)
        if self.do_random_rotation:
            x = inverse_randomized_hadamard_transform(z.float(), prng.manual_seed(self.seed), same_rotation_batch) / self.s
        else:
            x = z.float() / self.s
        self.seed = None
        return x[:, :self.d]

    def add_noise(self, grad_vec, same_rotation_batch, final_microbatch):
        d = grad_vec.size(1)
        grad_vec = self.decode(self.privatize(grad_vec, same_rotation_batch=same_rotation_batch, final_microbatch=final_microbatch), same_rotation_batch=same_rotation_batch)
        return grad_vec


def binary_search(func, constraint, minimum, maximum, tol=1e-5):
    """
    Performs binary search on monotonically increasing function `func` between
    `minimum` and `maximum` to find the maximum value for which the function's
    output satisfies the specified `constraint` (which is a binary function).
    Returns maximum value `x` at which `constraint(func(x))` is `True`.

    The function takes an optional parameter specifying the tolerance `tol`.
    """
    assert constraint(func(minimum)), "constraint on function must hold at minimum"

    # evaluate function at maximum:
    if constraint(func(maximum)):
        return maximum
    # perform the binary search:
    while maximum - minimum > tol:
        midpoint = (minimum + maximum) / 2.
        if constraint(func(midpoint)):
            minimum = midpoint
        else:
            maximum = midpoint
    return minimum

def optimal_scaling_integer(d, l2_norm_bound, beta, tol=1e-3):
    def constraint(t):
        if t == 0:
            return True
        quantized_norm = post_rounding_l2_norm_bound(d, t, beta)
        return quantized_norm <= l2_norm_bound + 1e-6
    opt_norm = binary_search(lambda t: t, constraint, 0, l2_norm_bound, tol=tol)
    return opt_norm / l2_norm_bound

def post_rounding_l2_norm_bound(d, l2_norm_bound, beta=np.exp(-.5)):
    """
    Function for computing vector norm bound after quantizing to the integer grid.
    Adapted from https://github.com/google-research/federated/blob/master/distributed_dp/compression_utils.py
    """
    bound1 = l2_norm_bound + math.sqrt(d)
    squared_bound2 = l2_norm_bound**2 + 0.25 * d
    squared_bound2 += (math.sqrt(2.0 * math.log(1.0 / beta)) * (l2_norm_bound + 0.5 * math.sqrt(d)))
    bound2 = math.sqrt(squared_bound2)
    # bound2 is inf if beta = 0, in which case we fall back to bound1.
    return min(bound1, bound2)

def clip_gradient(norm_clip, linf_clip, grad_vec, p=2, small_constant=1e-5):
        """
        Lp norm clip to norm_clip and then L-inf norm clip to linf_clip.
        Args:
            norm_clip: Lp norm clip
            linf_clip: L-inf norm clip
            grad_vec: gradient vector
            p: Lp norm
            small_constant: small constant to avoid numerical issues
        """
        C = norm_clip * (1 - small_constant)
        if len(grad_vec.shape) == 1:
            grad_norm = torch.linalg.vector_norm(grad_vec, ord=p, dim=None, keepdim=False, dtype=torch.float32)
            if grad_norm > C:
                grad_vec *= C / grad_norm
        else:
            grad_norm = torch.linalg.vector_norm(grad_vec, ord=p, dim=1, keepdim=False, dtype=torch.float32)
            multiplier = torch.ones_like(grad_norm, dtype=torch.float32)
            multiplier[grad_norm.gt(C)] = C / grad_norm[grad_norm.gt(C)]
            grad_vec *= multiplier.unsqueeze(1)
        if linf_clip > 0: # can do accounting based only on l2 clipped norm
            grad_vec.clamp_(-linf_clip, linf_clip)

        return grad_vec

def params_to_vec(model, return_type="param"):
    '''
    Helper function that concatenates model parameters or gradients into a single vector.
    '''
    vec = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if return_type == "param":
            vec.append(param.data.view(1, -1))
        elif return_type == "grad":
            vec.append(param.grad.view(1, -1))
        elif return_type == "grad_sample":
            if hasattr(param, "grad_sample"):
                vec.append(param.grad_sample.view(param.grad_sample.size(0), -1))
            else:
                print("Error from Skellam: Per-sample gradient not found")
                sys.exit(1)
        else:
            raise ValueError(f"Invalid return type in Skellam: {return_type}")
    return torch.cat(vec, dim=1).squeeze()

def set_grad_to_vec(model, vec):
    '''
    Helper function that sets the model's gradient to a given vector.
    '''
    model.zero_grad()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        size = param.data.view(1, -1).size(1)
        param.grad = vec[:size].view_as(param.data).clone()
        vec = vec[size:]
    return
