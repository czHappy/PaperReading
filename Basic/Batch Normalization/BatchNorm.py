# Created by czHappy at 2019/3/21
import time
import torch
import torchvision
import torch.nn as nn
import sys
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
class FlattenLayer(nn.Module):  # 自己定义层Flattenlayer
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1) #gamma beta的维度跟通道数强相关 其他都是1
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
#定义网络
net = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            FlattenLayer(),
            nn.Linear(16*4*4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
net = net.to(device)

#get Data
batch_size = 256
#transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',download=True,
                                              train=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',download=True,
                                             train=False, transform=transform)
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

lr, num_epochs = 0.001, 20
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# evaluate_accuracy
def evaluate_accuracy(test_iterator, net):
    with torch.no_grad():
        device = list(net.parameters())[0].device
        test_acc_sum = 0.0
        ncount = 0
        for x_test, y_test in test_iterator:
            if isinstance(net, torch.nn.Module):
                net.eval()
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_hat = net(x_test)
                test_acc_sum += (y_hat.argmax(dim=1) == y_test).sum().cpu().item()
                ncount+=len(y_test)
                net.train()
        test_acc = test_acc_sum/ncount
        return test_acc
def train(num_epoch):
    for epoch in range(num_epoch):
        l_sum, train_acc_sum, ncount, start = 0.0, 0.0, 0, time.time()
        for x_train, y_train in train_iter:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_hat = net(x_train)
            l = loss(y_hat, y_train)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y_train).sum().cpu().item()
            ncount += y_train.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f , spend_time: %.4f' %
              (epoch+1, l_sum/ncount,train_acc_sum/ncount, test_acc,time.time()-start))


if __name__ == "__main__":
    train(num_epochs)
# train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
# with BN
# epoch: 1, train_loss: 0.0039, train_acc: 0.7853, test_acc: 0.8369 , spend_time: 14.0933
# epoch: 2, train_loss: 0.0018, train_acc: 0.8636, test_acc: 0.8583 , spend_time: 13.8332
# epoch: 3, train_loss: 0.0014, train_acc: 0.8796, test_acc: 0.8711 , spend_time: 14.0801
# epoch: 4, train_loss: 0.0013, train_acc: 0.8871, test_acc: 0.8641 , spend_time: 14.5470
# epoch: 5, train_loss: 0.0012, train_acc: 0.8935, test_acc: 0.8732 , spend_time: 15.8111
# epoch: 6, train_loss: 0.0011, train_acc: 0.8978, test_acc: 0.8667 , spend_time: 14.4228
# epoch: 7, train_loss: 0.0011, train_acc: 0.9016, test_acc: 0.8432 , spend_time: 14.4978
# epoch: 8, train_loss: 0.0011, train_acc: 0.9043, test_acc: 0.8797 , spend_time: 15.1705
# epoch: 9, train_loss: 0.0010, train_acc: 0.9075, test_acc: 0.8660 , spend_time: 13.9636
# epoch: 10, train_loss: 0.0010, train_acc: 0.9107, test_acc: 0.8800 , spend_time: 14.0225
# epoch: 11, train_loss: 0.0010, train_acc: 0.9137, test_acc: 0.8735 , spend_time: 13.4804
# epoch: 12, train_loss: 0.0009, train_acc: 0.9153, test_acc: 0.8652 , spend_time: 13.4552
# epoch: 13, train_loss: 0.0009, train_acc: 0.9171, test_acc: 0.8653 , spend_time: 13.5266
# epoch: 14, train_loss: 0.0009, train_acc: 0.9195, test_acc: 0.8854 , spend_time: 13.4589
# epoch: 15, train_loss: 0.0009, train_acc: 0.9214, test_acc: 0.8857 , spend_time: 13.9330
# epoch: 16, train_loss: 0.0008, train_acc: 0.9230, test_acc: 0.8812 , spend_time: 13.6732
# epoch: 17, train_loss: 0.0008, train_acc: 0.9249, test_acc: 0.8922 , spend_time: 13.6339
# epoch: 18, train_loss: 0.0008, train_acc: 0.9285, test_acc: 0.8630 , spend_time: 13.4923
# epoch: 19, train_loss: 0.0008, train_acc: 0.9287, test_acc: 0.8814 , spend_time: 13.5800
# epoch: 20, train_loss: 0.0007, train_acc: 0.9308, test_acc: 0.8896 , spend_time: 13.4751


#without BN
# epoch: 1, train_loss: 0.0075, train_acc: 0.2923, test_acc: 0.5655 , spend_time: 11.0225
# epoch: 2, train_loss: 0.0039, train_acc: 0.6101, test_acc: 0.6599 , spend_time: 10.8968
# epoch: 3, train_loss: 0.0032, train_acc: 0.6996, test_acc: 0.7246 , spend_time: 11.0025
# epoch: 4, train_loss: 0.0028, train_acc: 0.7348, test_acc: 0.7387 , spend_time: 11.0342
# epoch: 5, train_loss: 0.0026, train_acc: 0.7504, test_acc: 0.7538 , spend_time: 10.9370
# epoch: 6, train_loss: 0.0024, train_acc: 0.7639, test_acc: 0.7616 , spend_time: 10.9174
# epoch: 7, train_loss: 0.0022, train_acc: 0.7749, test_acc: 0.7740 , spend_time: 10.9114
# epoch: 8, train_loss: 0.0021, train_acc: 0.7855, test_acc: 0.7795 , spend_time: 10.9031
# epoch: 9, train_loss: 0.0021, train_acc: 0.7947, test_acc: 0.7901 , spend_time: 10.9791
# epoch: 10, train_loss: 0.0020, train_acc: 0.8027, test_acc: 0.7912 , spend_time: 11.1465
# epoch: 11, train_loss: 0.0019, train_acc: 0.8098, test_acc: 0.8035 , spend_time: 10.9123
# epoch: 12, train_loss: 0.0019, train_acc: 0.8156, test_acc: 0.8103 , spend_time: 11.0491
# epoch: 13, train_loss: 0.0018, train_acc: 0.8213, test_acc: 0.8098 , spend_time: 10.9940
# epoch: 14, train_loss: 0.0018, train_acc: 0.8246, test_acc: 0.8167 , spend_time: 10.9449
# epoch: 15, train_loss: 0.0018, train_acc: 0.8292, test_acc: 0.8203 , spend_time: 11.0199
# epoch: 16, train_loss: 0.0017, train_acc: 0.8337, test_acc: 0.8238 , spend_time: 11.2112
# epoch: 17, train_loss: 0.0017, train_acc: 0.8360, test_acc: 0.8238 , spend_time: 11.1996
# epoch: 18, train_loss: 0.0017, train_acc: 0.8389, test_acc: 0.8262 , spend_time: 10.9224
# epoch: 19, train_loss: 0.0016, train_acc: 0.8407, test_acc: 0.8292 , spend_time: 10.9304
# epoch: 20, train_loss: 0.0016, train_acc: 0.8446, test_acc: 0.8316 , spend_time: 11.1930