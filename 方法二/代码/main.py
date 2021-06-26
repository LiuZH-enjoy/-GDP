import pandas as pd
import numpy as np
import lr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


# 读取数据
data = pd.read_csv('实验数据.csv',encoding='utf-8',header=None)

## 建立第一个二元回归：自变量为年份+国内生产总值，因变量为人均国内生产总值
model1 = lr.LinearRegression() # 建立模型
x_train_1 = data.iloc[:, [0,1]] # 自变量
y_train_1 = data.iloc[:,2] # 因变量
model1.fit_normal(x_train_1,y_train_1) # 模型拟合
x_pred_1 = np.array([2021,1082387.8745]).reshape(1,-1) # 传入需要预测的自变量
pre1 = model1.predict(x_pred_1) # 得到预测值
print(f'2021年人均国内生产总值预测值为：{pre1}亿元') # 打印输出预测值
## 画出拟合图
X1 = data.iloc[:,0] # 第一个自变量：年份
X2 = data.iloc[:,1] # 第二个自变量：国内生产总值
Y1 = data.iloc[:,2] # 因变量：人均国内生产总值
x1=np.linspace(2010,2021,12) # 设置x1坐标范围
x2=np.linspace(400000,1100000,12) # 设置x2坐标范围
fig1 = plt.figure() # 设置画布1
ax = Axes3D (fig1) # 3d图
x1, x2 = np.meshgrid(x1, x2) # 整合坐标为二维
ax.plot_surface(x1,x2,model1.interception_+model1.coef_[0]*x1+model1.coef_[1]*x2, alpha=0.5) # 画出拟合平面
ax.scatter(X1, X2, Y1, color='#ff0000') # 画出原数据样本散点


## 建立第二个二元回归
model2 = lr.LinearRegression()
x_train_2 = data.iloc[:,[0,2]]
y_train_2 = data.iloc[:,1]
model2.fit_normal(x_train_2, y_train_2)
x_pred_2 = np.array([2021, 77205.8909]).reshape(1,-1)
pre2 = model2.predict(x_pred_2)
print(f'2021年国内生产总值预测值为：{pre2}元')
## 画出拟合图，各代码含义同上
X1 = data.iloc[:,0]
X2 = data.iloc[:,2]
Y1 = data.iloc[:,2]
x1=np.linspace(2010,2021,12)
x2=np.linspace(30000,75000,12)
fig2 = plt.figure()
ax = Axes3D (fig2)
x1, x2 = np.meshgrid(x1, x2)
ax.plot_surface(x1,x2,model2.interception_+model2.coef_[0]*x1+model2.coef_[1]*x2, alpha=0.5)
ax.scatter(X1, X2, Y1, color='#ff0000')
plt.show() # 显示图片