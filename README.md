# Signum_pytorch
# This is a project for Signum optimizer implemented by Pytorch.
## see details in the original paper at: https://arxiv.org/abs/1711.05101

##The optimizer updates the weight by:
    buf = momentum * buf + (1-momentum)*rescaled_grad
    weight = (1 - lr * weight_decay) * weight - lr * sign(buf)
    
##This optimizer accepts the following parameters in addition to those accepted
  by :class:`.Optimizer`.
##Parameters
    ----------
    momentum : float, optional
       The momentum value.
    wd_lh : float, optional
       The amount of decoupled weight decay regularization
