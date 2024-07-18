# -*- coding:utf-8 -*-
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from IMATCN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, h_feats)

        '''创新A 定义更深的图卷积层'''
        self.conv3 = SAGEConv(h_feats, out_feats)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            x = F.relu(self.conv1(x, edge_index))
            '''创新A 应用更深的图卷积层'''
            x = F.relu(self.conv2(x, edge_index))
            # x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)
        else:
            x = F.relu(self.conv1(x, edge_index))
            '''创新A 应用更深的图卷积层'''
            x = F.relu(self.conv2(x, edge_index))
            # x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)

            '''创新B 在上面的代码中，将elu更换为了relu'''
        return x


# ######################################################################

'''创新 '''


class GraphSAGE_IMATCN(nn.Module):
    def __init__(self, args):
        super(GraphSAGE_IMATCN, self).__init__()
        self.args = args
        self.out_feats = 128
        self.sage = GraphSAGE(in_feats=args.seq_len, h_feats=128, out_feats=self.out_feats)
        OutputChannelList = [32]  # TCN各层输出维度
        n_heads = 4  # self-attention 头数, 需要可以整除tcn_OutputChannelList最后一个数字
        self.tcn = IMATCN(input_features_num=args.input_size, input_len=args.seq_len, output_len=128,
                          tcn_OutputChannelList=OutputChannelList, tcn_KernelSize=2,
                          tcn_Dropout=0.1, n_heads=n_heads)
        self.fcs = nn.ModuleList()
        for k in range(args.input_size):
            self.fcs.append(nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, args.output_size)
            ))

    def forward(self, data):
        # Data(x=[13, 24], edge_index=[2, 32], y=[13, 1])
        # DataBatch(x=[6656, 24], edge_index=[2, 16384], y=[6656, 1], batch=[6656], ptr=[513])
        # output(13, 512, 1) y(512, 13, 1)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = torch.max(batch).item() + 1
        x = self.sage(x, edge_index)  # 6656 128 = 512 * (13, 128)   # y = 6656 1 = 512 * (13 1)
        batch_list = batch.cpu().numpy()
        # print(batch_list)
        # split
        xs = [[] for k in range(batch_size)]
        ys = [[] for k in range(batch_size)]
        for k in range(x.shape[0]):
            xs[batch_list[k]].append(x[k, :])
            ys[batch_list[k]].append(data.y[k, :])

        xs = [torch.stack(x, dim=0) for x in xs]
        ys = [torch.stack(x, dim=0) for x in ys]
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        # print(x.shape, y.shape)  # 512 13 128 / 512 13 1
        # output(13, 512, 1) y(512, 13, 1)
        x = x.permute(0, 2, 1)  # 512 128 13
        x = self.tcn(x)
        # x = x[:, -1, :]
        preds = []
        for fc in self.fcs:
            preds.append(fc(x))

        pred = torch.stack(preds, dim=0)

        return pred, y

# ######################################################################
