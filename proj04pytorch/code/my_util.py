import numpy as np
import collections
from glob import glob
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
#from PIL import Image
#import torchvision.transforms as transforms
#import PIL
import matplotlib.pyplot as plt


def debug_img(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

def debug_images(dataloaders, image_datasets):
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes # copied from pytorch tutorial, but here it's number, not really name
    print("dataset sizes", dataset_sizes)
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    debug_img(out, title=[class_names[x] for x in classes])
    #imshow1(out, title=[x for x in classes])
    
def save_checkpoint(model, filename, debug, image_datasets):
    checkpoint={
        'state_dict': model.state_dict(),
        'debug':str(debug),
        'class_to_idx': image_datasets['train'].class_to_idx 
    }
    torch.save(checkpoint, filename)
    

def freeze_layers(model, firstLayersToFreeze):
    ct = 0
    for name, child in model.named_children():
        ct += 1
        if ct < firstLayersToFreeze:
            for name2, params in child.named_parameters():
                params.requires_grad = False

'''
def freeze_layers_all(model):
    for prm in model.parameters():
        prm.requires_grad = False
'''    

# partially used https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/  
def do_validation(model, valid_loader, criteria, pu):
    running_loss = 0
    running_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images=images.to(pu)
            labels=labels.to(pu)
            output = model.forward(images)
            loss = criteria(output, labels)
            _, predicts = torch.max(output.data, 1)
            running_loss += loss.item()* images.size(0)
            running_acc += (predicts == labels).sum().item()      
    return running_loss/len(valid_loader.dataset), running_acc/len(valid_loader.dataset)

def acc_is_better_than(acc1, acc2):
    if acc1['validAcc']> acc2['validAcc']:
        return True
    if acc1['validAcc'] < acc2['validAcc']:
        return False
    # if validAcc is the same, then look into trainValie:
    return acc1['trainAcc'] > acc2['trainAcc']
    
def do_training(epochs, model, optimizer, scheduler, criteria, dataloaders, pu, checkpoint_file, image_datasets):
    train_loader=dataloaders['train']
    valid_loader=dataloaders['valid']
    model.to(pu) 

    print('Starting train: pu={} epochs={}'.format(pu, epochs))      
    bestAcc={'validAcc':-1, 'trainAcc':-1}
    for ep in range(epochs):
        running_train_loss = 0
        running_train_acc=0
        scheduler.step() # new epoch, possible lr decay
        
        for idx, (images, labels) in enumerate(train_loader):
            #print('...starting epoch:{} step:{}'.format(ep, steps))
            images=images.to(pu)
            labels=labels.to(pu)
            model.train() # this affects things like dropout

            optimizer.zero_grad() # reset the gradient calc for this epoch
            outputs = model.forward(images)
            loss = criteria(outputs, labels)
            loss.backward()     # back propagation
            optimizer.step()    # weights fixing step
    
            # error calculations:
            _, predicts = torch.max(outputs.data, 1)
            running_train_loss += loss.item() * images.size(0)
            running_train_acc += (predicts == labels).sum().item()
        
        # after one epoc iteration on the entire data, collect TRAIn epoch stats:
        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = running_train_acc / len(train_loader.dataset)   
        valid_loss, valid_acc = do_validation(model, valid_loader, criteria, pu)           
        print('Epoch={} trainLoss={:.4f} trainAcc={:.4f} validLoss={:.4f} validAcc={:.4f}'.format(
            ep, train_loss, train_acc, valid_loss, valid_acc))
    
        currAcc={'trainAcc':train_acc, 'validAcc':valid_acc}
        if acc_is_better_than(currAcc, bestAcc):
            bestAcc= currAcc
            save_checkpoint(model, checkpoint_file, currAcc, image_datasets)
            
