
import torch
from torchvision import  models
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from ranger import Ranger  # this is from ranger.py
import time
import os
import shutil
from data_load import split_Train_Val_Data
from LabelSmoothing import LabelSmoothingCrossEntropy


data_dir = 'data'
# 读取数据
dataloders ,dataset_sizes = split_Train_Val_Data(data_dir,(0.9,0.1),batch_size=8)

use_gpu = torch.cuda.is_available()
print('use cuda')
def train_model(model, lossfunc, optimizer, scheduler, num_epochs=10):
    start_time = time.time()
    elapsed_time = 0

    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc = []
    valid_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = lossfunc(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':
                valid_acc.append(epoch_acc)
            else:
                train_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        epch_model_name ='checkpiont/ep{}_train{}_val{}.pth'.format(epoch,train_acc[-1],valid_acc[-1],)

        torch.save(model.state_dict(), epch_model_name)
        print('model saved : ',epch_model_name)
        elapsed_time = time.time() - start_time - elapsed_time
        print('epoch complete in {:.0f}m {:.0f}s'.format(
            elapsed_time // 60, elapsed_time % 60))
        # 这里使用了学习率调整策略
        scheduler.step(valid_acc[-1])
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_acc,valid_acc

if os.path.exists('checkpiont')==False:
    os.mkdir('checkpiont')
else:
    shutil.rmtree('checkpiont')
    os.mkdir('checkpiont')
# 模型训练
model_ft = models.mobilenet_v2(pretrained=False)
model_ft.load_state_dict(torch.load('./premodel/mobilenet_v2-b0353104.pth'))
# in_features=1280
# print(model_ft.classifier)
model_ft.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(1280, 6))
# print(model_ft.classifier)
if use_gpu:
    model_ft = model_ft.cuda()

# define loss function
# lossfunc = nn.CrossEntropyLoss()
lossfunc = LabelSmoothingCrossEntropy()

parameters = list(model_ft.parameters())
# optimizer_ft = optim.SGD(parameters, lr=0.001, momentum=0.9, nesterov=True)

# 使用Ranger优化器
optimizer_ft = Ranger(parameters, 0.001,weight_decay=0)

# 使用ReduceLROnPlateau学习调度器，如果2个epoch准确率没有提升，则减少学习率
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='max',patience=2,verbose=True)
torch.cuda.empty_cache()
model_ft,train_acc,valid_acc = train_model(model=model_ft,
                           lossfunc=lossfunc,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=5)
torch.save(model_ft.state_dict(), './model_out/bestmodel.pth')
