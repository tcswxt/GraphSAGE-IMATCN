# -*- coding:utf-8 -*-
import copy
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from get_data import setup_seed
from models import GraphSAGE_IMATCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_seed(123)

def get_val_loss(args, model, Val):

    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for graph in Val:
        graph = graph.to(device)
        preds, labels = model(graph)
        total_loss = 0
        for k in range(args.input_size):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        total_loss /= preds.shape[0]

        val_loss.append(total_loss.item())

    return np.mean(val_loss)

def train(args, Dtr, Val, model_type, path):

    model = GraphSAGE_IMATCN(args).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for graph in Dtr:
            graph = graph.to(device)
            preds, labels = model(graph)
            total_loss = 0
            for k in range(args.input_size):
                total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])

            total_loss = total_loss / preds.shape[0]
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.copy(model)
            state = {'model': best_model.state_dict()}
            torch.save(state, path + '/GraphSAGE-IMATCN/models/' + model_type + '.pkl')

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict()}
    torch.save(state, path + '/GraphSAGE-IMATCN/models/' + model_type + '.pkl')

# ########################################################################################
