# -*- coding:utf-8 -*-
import argparse
import torch

def GraphSAGE_IMATCN_args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--input_size', type=int, default=8, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=128, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--num_channels', type=list, default=[32, 32], help='num_channels')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=150, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args
