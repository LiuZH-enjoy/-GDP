import random
import torch
import pandas as pd
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
data = pd.read_csv('实验数据.csv',encoding='utf-8',header=None)
features = torch.Tensor(data.iloc[:,0]).unsqueeze(1)/100
labels_1 = torch.Tensor(data.iloc[:,1])/10000
labels_2 = torch.Tensor(data.iloc[:,2])/10000

# 定义数据生产器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w)+b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

# 定义优化器，这里采用SGD
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

# 参数设置与初始化
lr = 0.001
num_epochs = 100
batch_size = 5
net = linreg
loss = squared_loss
w = torch.ones(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# 第一个拟合模型训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels_1):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        if epoch == num_epochs-1:
            print('k值为：',w.detach().numpy())
            print('b值为：',b.detach().numpy())
        train_l = loss(net(features, w, b), labels_1)
        #print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')

print(f'2021年国内生产总值预测值为：{(w.detach().numpy()*20.21+b.detach().numpy())*10000}亿元')

# 画出拟合图
plt.figure(1)
plt.scatter(features[:, (0)].detach().numpy(), labels_1.detach().numpy())
plt.grid()
x1=np.linspace(2010,2021,12)/100
y1=(w.detach().numpy()*(x1)+b.detach().numpy())
plt.plot(x1,y1)


# 参数设置与初始化
lr = 0.001
num_epochs = 100
batch_size = 5
net = linreg
loss = squared_loss
w = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# 第二个拟合模型训练
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels_2):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels_2)
        if epoch == num_epochs-1:
            print('k值为：',w.detach().numpy())
            print('b值为：',b.detach().numpy())
        #print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')

# 画出拟合图
print(f'2021年人均国内生产总值为：{(w.detach().numpy()*20.21+b.detach().numpy())*10000}元')
plt.figure(2)
plt.plot()#画在图2上，且不在一个窗口
plt.scatter(features[:, (0)].detach().numpy(), labels_2.detach().numpy())
plt.grid()
x1=np.linspace(2010,2021,12)/100
y1=(w.detach().numpy()*(x1)+b.detach().numpy())
plt.plot(x1,y1)
plt.show()