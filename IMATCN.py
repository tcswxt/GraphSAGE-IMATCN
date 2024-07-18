import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-2):
        super(FRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((1, num_features[0], 1)))
        if isinstance(num_features, list):
            num_features = num_features[0]
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1)))

    def forward(self, x):
        # Calculate mean and variance along the channel dimension
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x

# TCN-attention
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.4):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()

        ''' 创新F     批标准化的定义  '''
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
        # self.bn1 = nn.BatchNorm1d(n_inputs)

        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len

        self.PRelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        ''' 创新F     批标准化的定义 '''
        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
        # self.bn2 = nn.BatchNorm1d(n_outputs)

        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len

        self.PRelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        ''' 使用批标准化 '''
        self.net = nn.Sequential(self.conv1, self.chomp1, self.PRelu1, self.dropout1,
                                 self.conv2, self.chomp2, self.PRelu2, self.dropout2)
        # self.net = nn.Sequential(self.bn1, self.conv1, self.chomp1, self.PRelu1, self.dropout1,
        #                          self.bn2, self.conv2, self.chomp2, self.PRelu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        '''   更换激活函数   '''
        # self.relu = nn.ReLU()
        self.PRelu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化
        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.PRelu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.4):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            layers += [nn.LeakyReLU()]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class TCNs(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNs, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        '''创新G   '''
        self.frn = FRN(num_channels)

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = inputs.transpose(1, 2)
        y1 = self.tcn(inputs)

        '''创新G   '''
        y1 = self.frn(y1)

        return y1


class IMATCN(nn.Module):
    def __init__(self, input_features_num, input_len, output_len, tcn_OutputChannelList,
                 tcn_KernelSize, tcn_Dropout, n_heads):
        super(IMATCN, self).__init__()
        self.tcnunit = TCNs(input_features_num, tcn_OutputChannelList, tcn_KernelSize, tcn_Dropout)
        self.attentionunit = nn.MultiheadAttention(tcn_OutputChannelList[-1], n_heads)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(tcn_OutputChannelList[-1] * input_len, output_len)

    def forward(self, input_seq):
        tcn_out = self.tcnunit(input_seq)
        tcn_out = tcn_out.permute((0, 2, 1))
        att_out, att_weights = self.attentionunit(tcn_out, tcn_out, tcn_out)
        att_out += tcn_out
        flatten_out = self.flatten(att_out)
        fc_out = self.linear(flatten_out)
        return fc_out
