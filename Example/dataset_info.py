#coding=utf-8


from datasets import indoor


def load_info(datasets_dir):
    CLASS_NUM = 67
    trainset = indoor.Dataset(datasets_dir, train=True)
    testset = indoor.Dataset(datasets_dir, train=False, test=True)

    return  CLASS_NUM,trainset,testset