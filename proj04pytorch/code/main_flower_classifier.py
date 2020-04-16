import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from my_util import debug_images
from my_util import do_training
from my_util import do_validation
from my_util import freeze_layers
from my_util import save_checkpoint

# Trains a CNN and saves it to file.
# This code was part of my submission for "Facebook PyTorch Challenge" as taught in the course.
# Please check the constants below and adjust.

# Various resources and constants.
# Please note:
# 1. Change path to match the directory structure on your machine.
# 2. REMINDER: the small dataset "flower_data_small" will produce low-quality results. Download the full 300MB !
resource_dir='C:/pel/workspaces/python-ws/proj04pytorch/resources'
data_dir = f'{resource_dir}/flower_data_small'
train_dir = f'{data_dir}/train'
valid_dir = f'{data_dir}/valid'
cat_to_name_json_file = f'{resource_dir}/cat_to_name.json'
batch_size=32

lr = 0.003
lr_step = 7 # decary the lr every X number of epochs
lr_gamma=0.1
momentum=0.9
epochs = 2
# dropout=0.5 # Use dropouts if necessary, e.g. to prevent overefitting
pu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criteria= nn.CrossEntropyLoss()
#Explore other options such as: criteria = nn.NLLLoss() # matches softmax according to lecture
frozen_layers_num=30
checkpoint_file='checkpoint.pth'

# Defining transforms for the training and validation sets
# I used article https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomRotation(45),              
        #transforms.RandomResizedCrop(224),       
        #transforms.RandomHorizontalFlip(),          
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),   
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# Loading datasets using ImageFolder:
image_datasets = {
    'train': datasets.ImageFolder(root=train_dir,transform=data_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dir,transform=data_transforms['valid']) 
}

# dataloaders, corresponding to the above image sets:
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=batch_size)
    #'test' : DataLoader(image_datasets['test'],batch_size = batch_size)    
}

# Debugging some info on the loaded images:
debug_images(dataloaders, image_datasets)
print('= class to idx:\n')
for key,val in image_datasets['train'].class_to_idx.items():
    print(key, "-", val)

# Load the mapping from category label -> category name. 
# Reading the json using: https://docs.python.org/2/library/json.html
# This yields a dictionary mapping the integer encoded categories to names of the flowers.
with open(cat_to_name_json_file, 'r') as f:
    cat_to_name = json.load(f)

# ===========================
# main training execution!
# ===========================
def make_model():
    model1=models.resnet152(pretrained=True)
    # make sure the pretrained weights are left alone:
    freeze_layers(model1, frozen_layers_num )
    model1.fc = nn.Linear(model1.fc.in_features, 102)
    return model1

model=make_model()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)
# Explore other optimizers such as: optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer,lr_step, lr_gamma)
do_training(epochs, model, optimizer, scheduler, criteria, dataloaders, pu, checkpoint_file, image_datasets)