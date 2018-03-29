# Signum_pytorch
## This is the repository for Signum optimizer implemented by Pytorch.
### see details in the original paper at: https://arxiv.org/abs/1711.05101

Args:\
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups\
        lr (float): learning rate\
        momentum (float, optional): momentum factor (default: 0.9)\
        weight_decay (float, optional): weight decay (default: 0)

    Example:
        >>> optimizer = signum.Signum(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

Note:\
        The optimizer updates the weight by:\
            buf = momentum * buf + (1-momentum)*rescaled_grad\
            weight = (1 - lr * weight_decay) * weight - lr * sign(buf)

Considering the specific case of Momentum, the update Signum can be written as

        .. math::
                 \begin{split}g_t = \nabla J(W_{t-1})\\
                 m_t = \beta m_{t-1} + (1 - \beta) g_t\\
		 W_t = W_{t-1} - \eta_t \text{sign}(m_t)}\end{split}

If do not consider Momentum, the update Sigsgd can be written as

        .. math::
            	 g_t = \nabla J(W_{t-1})\\
                 W_t = W_{t-1} - \eta_t \text{sign}(g_t)}
