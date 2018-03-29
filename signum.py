#coding=utf-8
import torch
from torch.optim import Optimizer

class Signum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.09, wd_lh = 0.0, dampening=0, weight_decay = 0, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, wd_lh=wd_lh,dampening =dampening,
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
            dampening = group['dampening']
            wd_lh = group['wd_lh']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state: #signsgd
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                        d_p = torch.sign(buf)
                    else:  #signum
                        buf = param_state['momentum_buffer']
                        m = buf.clone()
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.mul_(1-wd_lh).add_(wd_lh,m)
                        d_p = torch.sign(buf)

                p.data.add_(-group['lr'], d_p)

        return loss
