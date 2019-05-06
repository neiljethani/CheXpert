#set path for packages
import sys
sys.path.insert(0,"/work/pip")

# torch
import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

# model
from models.densenet121 import make_model

# project
from train import train, validation
from Dataset import train_transform, validation_transform, CheXpertDataset

from utils.config_utils import load_config
from utils.fs_utils import create_folder
from utils.Timer import Timer

# system
import argparse
import time
import pickle

# create folder to store checkpoints
create_folder('checkpoints')
checkPath = 'checkpoints/session_' + Timer.timeFilenameString()
create_folder(checkPath)

# create folder to store best models
create_folder("best_models")
bestPath = "best_models/session_" + Timer.timeFilenameString()
create_folder(bestPath)

# create folder to store models for each epoch
create_folder("models")
modelPath = "models/session_" + Timer.timeFilenameString()
create_folder(modelPath)

# create folder to store log files
create_folder('logs')
logPath = 'logs/log_' + Timer.timeFilenameString()

# create folder to store history dictionary
create_folder("history")
hist_file = "history/session_" + Timer.timeFilenameString() + ".pkl"

def append_line_to_log(line = '\n'):
    """
    Append line into the log files.
    """
    with open(logPath, 'a') as f:
        f.write(line + '\n')

def parse_cli(params):
    """
    Parse the arguments for training the model.
    Args:
        - params: parameters read from "config.yaml"
    Returns:
        - args: arguments parsed by parser    
    """
    parser = argparse.ArgumentParser(description="PyTorch Transformer")
    
    # training 
    parser.add_argument('--batch-size', type=int, default=params['batch_size'], metavar='BZ',
                        help='input batch size for training (default: ' + str(params['batch_size']) + ')')
    
    parser.add_argument('--epochs', type=int, default=params['epochs'], metavar='EP',
                        help='number of epochs to train (default: ' + str(params['epochs']) + ')')
    
    # model
    parser.add_argument('--pretrained', type=bool, default=params['pretrained'], metavar='PT',
                        help='whether to use the pretrained weight (default: ' + str(params['pretrained']) + ')')
    parser.add_argument('--fixed', type=bool, default=params['fixed'], metavar='F',
                        help='whether to fix conv layers in pretrained model (default: ' + str(params['fixed']) + ')')

    # hyperparameters
    parser.add_argument('--lr', type=float, default=params['init_learning_rate'], metavar='LR',
                        help='inital learning rate (default: ' + str(params['init_learning_rate']) + ')')

    parser.add_argument('--beta1', type=float, default=params['beta1'], metavar='B1',
                        help=' Adam parameter beta1 (default: ' + str(params['beta1']) + ')')

    parser.add_argument('--beta2', type=float, default=params['beta2'], metavar='B2',
                        help=' Adam parameter beta2 (default: ' + str(params['beta2']) + ')')                    
                        
    parser.add_argument('--epsilon', type=float, default=params['epsilon'], metavar='EL',
                        help=' Adam regularization parameter (default: ' + str(params['epsilon']) + ')')

    parser.add_argument('--seed', type=int, default=params['seed'], metavar='S',
                        help='random seed (default: ' + str(params['seed']) + ')')  

    # system training
    parser.add_argument('--log-interval', type=int, default=params['log_interval'], metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--workers', type=int, default=0, metavar='W',
                        help='workers (default: 0)')

    parser.add_argument('--train_dir', default="./data/", type=str, metavar='PATHT',
                        help='path to the training files (default: data folder)')

    parser.add_argument('--val_dir', default="./data/", type=str, metavar='PATHV',
                        help='path to the validation files (default: data folder)')   

    args = parser.parse_args()

    return args     

############################### Main #################################################################

# to get the arguments
params = load_config('config.yaml')
args = parse_cli(params)

# to make everytime the randomization the same
torch.manual_seed(args.seed)

# to get the directory of training and validation
train_dir = args.train_dir
val_dir = args.val_dir

# to define training and validation dataloader
# Note to change the parser!!!!!
train_loader = torch.utils.data.DataLoader(CheXpertDataset(train_dir+"CheXpert-v1.0-small/train_preprocessed.csv", train_dir, transform=train_transform), batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(CheXpertDataset(val_dir+"CheXpert-v1.0-small/valid.csv", val_dir, transform=validation_transform), batch_size=args.batch_size, shuffle=False)    
                                                         
# to make the model
model = make_model(pretrained=args.pretrained, fixed=args.fixed)

# put model into the correspoinding device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# define optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

# define scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)

# define the criterion
# Note to add the classes weights!!!!!!!!!!!!
# HOW TO COMPUTE THE WEIGHT:
# |N|+|P|+|U| / |T|, where |T| = |N|, |P| or |U|
# the order in the following is N, P, U
criterion_no_finding = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.11, 9.98]).to(device))
criterion_en_card = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.12, 20.69, 18.01]).to(device))
criterion_card = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.19, 8.27, 27.63]).to(device))
criterion_lung_op = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.99, 2.12, 39.91]).to(device))
criterion_lung_le = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.05, 24.32, 150.14]).to(device))
criterion_edema = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.41, 4.28, 17.21]).to(device))
criterion_cons = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.24, 15.11, 8.05]).to(device))
criterion_pneu = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.12, 37.00, 11.90]).to(device))
criterion_atelec = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.43, 6.69, 6.62]).to(device))
criterion_pneurax = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.11, 11.49, 71.04]).to(device))
criterion_ple_eff = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.78, 2.59, 19.21]).to(device))
criterion_ple_other = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.03, 63.42, 84.21]).to(device))
criterion_frac = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.05, 24.71, 348.00]).to(device))
criterion_sup_dev = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.10, 1.93, 207.06]).to(device))

# construct a dictionary to store all the criterions
criterions = {
    "No Finding" : criterion_no_finding, 
    "Enlarged Cardiomediastinum" : criterion_en_card, 
    "Cardiomegaly" : criterion_card,
    "Lung Opacity" : criterion_lung_op,
    "Lung Lesion" : criterion_lung_le,
    "Edema" : criterion_edema,
    "Consolidation" : criterion_cons,
    "Pneumonia" : criterion_pneu,
    "Atelectasis" : criterion_atelec,
    "Pneumothorax" : criterion_pneurax,
    "Pleural Effusion" : criterion_ple_eff,
    "Pleural Other" : criterion_ple_other,
    "Fracture" : criterion_frac,
    "Support Devices" : criterion_sup_dev
}

# write the initial information to the log file
append_line_to_log("executing on device: ")
append_line_to_log(str(device))

# to use the inbuilt cudnn auto-tuner to to the best algorithm to use for the hardware
torch.backends.cudnn.benchmard = True

# to construct dictionary to store the training history
history = {"train_loss":[], "train_acc":[], "valid_loss":[], "valid_acc":[]}

# to set the best validation loss as inifinity
best_val_loss = np.inf

# training process
start_epoch = 1
for epoch in range(start_epoch, args.epochs + 1):
    # train
    train_loss, train_acc = train(epoch, model, optimizer, scheduler, criterions, train_loader, device, args.log_interval, append_line_to_log, checkPath)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    # validation
    valid_loss, valid_acc = validation(epoch, model, criterions, valid_loader, device, append_line_to_log)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)

    scheduler.step(valid_loss)

    # save the best model
    is_best = valid_loss < best_val_loss
    best_val_loss = min(valid_loss, best_val_loss)

    if is_best:
        best_model_file = "/best_model_" + str(epoch) + ".pth"
        best_model_file = bestPath + best_model_file
        torch.save(model.state_dict(), best_model_file)

    # save the model of this epoch
    model_file = "/model_" + str(epoch) + ".pth"
    model_file = modelPath + model_file

    torch.save(model.state_dict(), model_file)
    append_line_to_log("Save model to " + model_file)

# write the history dictionary to the pickle file
with open(hist_file, "wb") as fout:
    pickle.dump(history, fout)    
