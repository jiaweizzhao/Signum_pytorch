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
import dataset_info,train_info
import os
from datasets import VOC12_full,Caltech,indoor
import signum

EXPERIMENT_ID = ''


# Parameter Setting
BATCH_SIZE,NUM_EPOCHS,BATCHS_EPOCHS,INIT_LR,MOMENTUM,LR_DECAY_EPOCH,SELECTION_TYPE,COMBINE_PAR = train_info.load_info()

USE_GPU = torch.cuda.is_available()

# selection setting
datasets_dir = 'D:\Jiawei\ExperimentADMA\datasets'
TEST_BATCH_SIZE = 400
TEST_BATCHS = 10
PREDICT_BATCH_SIZE = TEST_BATCH_SIZE
CLASS_NUM,trainset,testset = dataset_info.load_info(datasets_dir)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False,num_workers=0) 
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0)
# Process
# model define
def train_model(model, criterion, optimizer, lr_scheduler, String):
    since = time.time()  

    best_model = model  
    best_acc = 0.0  
    num_epochs = NUM_EPOCHS

    # analysis 
    analysis_data = np.array([])
    analysis_auc = np.array([])

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  
        print('-' * 10) 
  
        # Each epoch has a training and validation

        # training
        optimizer = lr_scheduler(optimizer, epoch) 
        model.train(True)

        running_loss = 0.0  
        running_corrects = 0


        if epoch != 0:

            for batch in range(BATCHS_EPOCHS):
                for data in trainloader:

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
        else:
            analysis_data = np.append(analysis_data,0)

        # validation

        running_loss = 0.0  
        running_corrects = 0  

        model.train(False)


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

# learning rate define   
def exp_lr_scheduler(optimizer, epoch, init_lr=INIT_LR, lr_decay_epoch=LR_DECAY_EPOCH):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))  
  
    if epoch % lr_decay_epoch == 0:  
        print('LR is set to {}'.format(lr))  
  
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr  
  
    return optimizer

# Process
def train_process(String):
    
    print('Experiment:  ' + String + '  processing')
    # Pre-trained Model 
    # Resnet Model
    model_ft = models.resnet18(pretrained=True)
    model_ft.fc = nn.Linear(in_features=512, out_features=CLASS_NUM)


    if USE_GPU:  
        model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized  
    
    optimizer_ft = signum.Signum(model_ft.parameters(), lr=INIT_LR, momentum=MOMENTUM)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, String)
    
    # torch.save(model_ft, 'model_ft.pkl')


start_total = time.time()
train_process(EXPERIMENT_ID)
print((time.time()-start_total)/3600)