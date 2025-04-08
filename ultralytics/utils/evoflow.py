import torch
from torch.optim import Optimizer

class EvoFlow(Optimizer):
    def __init__(self, params, lr=None, beta1=None, beta2=None, alpha=None, lambda_=None, 
                 eps=None, evolve_freq=None, loss_fn=None):
        # Default values if not provided
        lr = lr if lr is not None else 0.001
        beta1 = beta1 if beta1 is not None else 0.87
        beta2 = beta2 if beta2 is not None else 0.999
        alpha = alpha if alpha is not None else 0.92
        lambda_ = lambda_ if lambda_ is not None else 0.002
        eps = eps if eps is not None else 1e-7

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, alpha=alpha, lambda_=lambda_, 
                        eps=eps)
        super(EvoFlow, self).__init__(params, defaults)
        self.evolve_freq = evolve_freq if evolve_freq is not None else 75
        self.loss_fn = loss_fn
        self.t = 0

    def step(self):
        self.t += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)  # Clip gradients to a maximum norm of 1.0
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['f'] = torch.zeros_like(p.data)
                    state['g_prev'] = torch.zeros_like(p.data)

                state['step'] += 1
                m, v, f, g_prev = state['m'], state['v'], state['f'], state['g_prev']
                beta1, beta2, alpha, epsilon = group['beta1'], group['beta2'], group['alpha'], group['eps']

                # Gradient consistency adjustment
                grad_consistency = torch.sum(grad * g_prev) / (torch.norm(grad) * torch.norm(g_prev) + epsilon)
                beta1_t = beta1 * max(0.65, grad_consistency.item())
                alpha_t = alpha * min(1.0, grad_consistency.item() + 0.65)

                # Update moving averages
                m.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                f.mul_(alpha_t).add_(grad, alpha=1 - alpha_t)
                v_hat = v / (1 - beta2 ** state['step'])
                c_t = 0.7 + 0.5 * torch.tanh(torch.abs(grad - g_prev) / (torch.sqrt(v_hat) + epsilon))

                # Evolutionary step
                if self.t % self.evolve_freq == 0:
                    p_data_old = p.data.clone()
                    grad_norm_old = torch.norm(grad)
                    delta = torch.normal(0, 0.01, p.data.shape, device=p.data.device)  # Increased perturbation magnitude
                    p.data.add_(delta)  # Apply perturbation
                    # Simulate a forward pass by checking the new gradient norm (heuristic)
                    grad_new = p.grad.data if p.grad is not None else grad
                    grad_norm_new = torch.norm(grad_new)
                    if grad_norm_new >= grad_norm_old:  # Revert if gradient norm increases
                        p.data.copy_(p_data_old)

                # Update parameters
                update = c_t * (m / (torch.sqrt(v_hat) + epsilon) + group['lambda_'] * p.data)
                p.data.add_(-group['lr'] * update)
                state['g_prev'].copy_(grad)