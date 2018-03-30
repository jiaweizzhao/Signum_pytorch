# Signum_pytorch
## This is the repository for Signum optimizer implemented in Pytorch.
### see the detailed discription of Signum in the original paper at: https://arxiv.org/abs/1711.05101

Arguments:\
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
            momentum = beta * momentum + (1-beta)*rescaled_grad\
            weight = (1 - lr * weight_decay) * weight - lr * sign(momentum)

Considering the specific case of Momentum, the update Signum can be written as

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20g_t%20%26%3D%20%5Cnabla%20J%28W_%7Bt-1%7D%29%5C%5C%20m_t%20%26%3D%20%5Cbeta%20m_%7Bt-1%7D%20&plus;%20%281%20-%20%5Cbeta%29%20g_t%5C%5C%20W_t%20%26%3D%20W_%7Bt-1%7D%20-%20%5Ceta_t%20%5Ctext%7Bsign%7D%28m_t%29%20%5Cend%7Balign*%7D)

If do not consider Momentum, the update Sigsgd can be written as

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20g_t%20%26%3D%20%5Cnabla%20J%28W_%7Bt-1%7D%29%5C%5C%20W_t%20%26%3D%20W_%7Bt-1%7D%20-%20%5Ceta_t%20%5Ctext%7Bsign%7D%28g_t%29%20%5Cend%7Balign*%7D)

Description of example:\
Using pre-trained resnet-18 model and supporting the datasets of Caltech101, VOC2012 and CVPR Indoor.\
Directory:\
train.py: train and validate the dataset\
signum.py: contain the signum optimizer\
dataset_info.py: load datasets information from datasets/ to train.py\
train_info.py: load training information to train.py\
datasets/: save the data pre-process for datasets
