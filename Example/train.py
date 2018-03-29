#!/usr/bin/env python3.6
'''
a demo for Signum implementation
you can directly see at line 227 -231
'''





import torch  
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable  
import numpy as np
from torchvision import  models, transforms
import time  
import copy
import selection # 选择
import analysis
import model_info,gpu_info,dataset_info,train_info
import os
from datasets import VOC12_aeroplane,VOC12_10,VOC12_bicycle,food_dataset,food_dataset10,VOC12_full,Caltech,indoor
import signum

EXPERIMENT_ID = ''

PC = 2172

# Parameter Setting
BATCH_SIZE,NUM_EPOCHS,BATCHS_EPOCHS,INIT_LR,MOMENTUM,LR_DECAY_EPOCH,SELECTION_TYPE,COMBINE_PAR = train_info.load_info()

USE_GPU = torch.cuda.is_available()

IF_FROZEN = False
FROZEN_NUM = model_info.load_frozen_num()



# selection setting
CHANGE_POOLING_EPOCHS = 1  # 8

datasets_dir,TEST_BATCH_SIZE,TEST_BATCHS = gpu_info.load_info()

PREDICT_BATCH_SIZE = TEST_BATCH_SIZE


CLASS_NUM,trainset,testset = dataset_info.load_info(datasets_dir)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0)

# Process
# model define
def train_model(model, criterion, optimizer, lr_scheduler, String):
    since = time.time()  

    best_model = model  
    best_acc = 0.0  
    num_epochs = NUM_EPOCHS

    # analysis 参数
    analysis_data = np.array([])
    analysis_auc = np.array([])

    selected_data = selecter.selection(model, 1, COMBINE_PAR)
    train_num = len(selected_data)
    selected_data = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=0)  # win修改 设置为0

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  
        print('-' * 10) 
  
        # Each epoch has a training and validation

        # training
        optimizer = lr_scheduler(optimizer, epoch)  # 再设置一下
        model.train(True)

        running_loss = 0.0  
        running_corrects = 0


        if epoch != 0:

            for batch in range(BATCHS_EPOCHS):
                for data in selected_data:

                    inputs, labels = data
                    if USE_GPU:
                        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (train_num * BATCHS_EPOCHS)
            epoch_acc = running_corrects / (train_num * BATCHS_EPOCHS)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

            analysis_data = np.append(analysis_data, BATCH_SIZE*(epoch+1)*BATCHS_EPOCHS)
            np.save("train_result_data/" + SELECTION_TYPE +"_SelectedData.npy", analysis_data)
            np.save("analysis/accuracy_analysis/" + SELECTION_TYPE + "_SelectedData.npy", analysis_data)
        else:
            analysis_data = np.append(analysis_data,0)
            np.save("train_result_data/" + SELECTION_TYPE +"_SelectedData.npy", analysis_data)
            np.save("analysis/accuracy_analysis/" + SELECTION_TYPE + "_SelectedData.npy", analysis_data)
        '''
        if epoch % 20 == 0 or epoch == NUM_EPOCHS-1:
            selecter.save_last_representation(model)
            print('save last representation finish!')
        '''

        # validation

        running_loss = 0.0  
        running_corrects = 0  

        model.train(False)

        # 计数器 减少验证数据量
        num = 0
        predict_class = []
        for data in testloader:
            
            if num >= TEST_BATCHS:
                break

            inputs, labels = data


            if USE_GPU:  
                inputs, labels = Variable(inputs, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
            else:  
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)

            data = outputs.data.cpu().numpy()

            for i in data:
                predict_class.append(i)

            _, preds = torch.max(outputs.data, 1)  
            loss = criterion(outputs, labels)

            running_loss += loss.data[0]  
            running_corrects += torch.sum(preds == labels.data)

            num = num + 1

        epoch_loss = running_loss / (TEST_BATCHS * TEST_BATCH_SIZE)
        epoch_acc = running_corrects / (TEST_BATCHS * TEST_BATCH_SIZE)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('validation', epoch_loss, epoch_acc))

        analysis_auc = np.append(analysis_auc, epoch_acc)
        np.save("train_result_data/val_predict_class.npy", np.array(predict_class))
        np.save("train_result_data/" + SELECTION_TYPE +"_Accuracy.npy", analysis_auc)
        np.save("analysis/accuracy_analysis/" + SELECTION_TYPE + "_Accuracy.npy", analysis_auc)
        analysis.auc_plot(analysis_data, analysis_auc, String)
        if epoch_acc > best_acc:  
            best_acc = epoch_acc  
            best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since  
    print('Training complete in {:.0f}m {:.0f}s'.format(  
        time_elapsed // 60, time_elapsed % 60))  
    print('Best val Acc: {:4f}'.format(best_acc)) 


    

    return best_model

# learning rate define    对于transfer learning 来说，学习率要小一些掌控大一些。
def exp_lr_scheduler(optimizer, epoch, init_lr=INIT_LR, lr_decay_epoch=LR_DECAY_EPOCH):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))  
  
    if epoch % lr_decay_epoch == 0:  
        print('LR is set to {}'.format(lr))  
  
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr  
  
    return optimizer

# Process
# 将训练过程包起来，可加相关改变参数
def train_process(String):
    
    print('Experiment:  ' + String + '  processing')
    # Pre-trained Model 引入
    # Alexmodel
    model_ft = model_info.load_model(CLASS_NUM)

    # 冻结参数改变
    if IF_FROZEN:
        num = FROZEN_NUM
        finetune_parameters_list = []
        for param in model_ft.parameters():
            if num == 0:
                finetune_parameters_list.append(param)
            else:
                param.requires_grad = False
                num -= 1
    if USE_GPU:  
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized  
    
    if IF_FROZEN:
        optimizer_ft = signum.Signum(finetune_parameters_list, lr=INIT_LR, momentum=MOMENTUM)
        # 只对分类器进行fine-tune 也可以自定义设置，见文档
    else:
        optimizer_ft = signum.Signum(model_ft.parameters(), lr=INIT_LR, momentum=MOMENTUM)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, String)
    
    save_setting(EXPERIMENT_ID)

    # torch.save(model_ft, 'model_ft.pkl')

def save_setting(EXPERIMENT_ID):

    setting_text = ['BATCH_SIZE: ' + str(BATCH_SIZE),
    'NUM_EPOCHS: ' + str(NUM_EPOCHS),
    'BATCHS_EPOCHS: ' + str(BATCHS_EPOCHS),
    'IF_FROZEN: ' + str(IF_FROZEN),
    'FROZEN_NUM: ' + str(FROZEN_NUM),
    'TEST_BATCH_SIZE: ' + str(TEST_BATCH_SIZE),
    'TEST_BATCHS: ' + str(TEST_BATCHS),
    'INIT_LR: '+ str(INIT_LR),
    'MOMENTUM: '+ str(MOMENTUM),
    'LR_DECAY_EPOCH: ' + str(LR_DECAY_EPOCH),
    'CLASS_NUM: ' + str(CLASS_NUM),
    'PREDICT_BATCH_SIZE: ' + str(PREDICT_BATCH_SIZE),
    'CHANGE_POOLING_EPOCHS: ' + str(CHANGE_POOLING_EPOCHS), '', '',
    'SELECTION_TYPE: ' + SELECTION_TYPE,
    'PREDICT_BATCH_SIZE: ' + str(PREDICT_BATCH_SIZE),
    'CHANGE_POOLING_EPOCHS: ' + str(CHANGE_POOLING_EPOCHS)
    ]
    file = open('Setting.txt','w')
    for textline in setting_text:
        file.write(textline)
        file.write('\n')



start_total = time.time()
train_process(EXPERIMENT_ID)
print((time.time()-start_total)/3600)