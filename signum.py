#coding=utf-8
import torch
from torch.optim import Optimizer


class Signum(Optimizer):
    """The Signum optimizer that takes the sign of gradient or momentum.

        The optimizer updates the weight by:

            rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
            buf = momentum * buf + (1-momentum)*rescaled_grad
            weight = (1 - lr * weight_decay) * weight - lr * sign(buf)
        see details in the original paper at:https://arxiv.org/abs/1711.05101
        This optimizer accepts the following parameters in addition to those accepted
        by :class:`.Optimizer`.

        Parameters
        ----------
        momentum : float, optional
           The momentum value.
        weight_decay : float, optional
           The amount of decoupled weight decay regularization,
    """
    def __init__(self, params, lr=0.01, momentum=0.09, weight_decay = 0, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        super(Signum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Signum, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    # signum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)

                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_((1 - momentum),d_p)
                    d_p = torch.sign(buf)
                else:#signsgd
                    d_p = torch.sign(d_p)

                p.data.add_(-group['lr'], d_p)

        return loss
