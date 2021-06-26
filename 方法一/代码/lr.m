function [y_2021,R_2] = lr(x,y)
% 计算数据量
n = size(x,1);
% 根据最小二乘法计算系数
k = (n*sum(x.*y)-sum(x)*sum(y))/(n*sum(x.*x)-sum(x)*sum(x));
% 根据最小二乘法计算偏置量
b = (sum(x.*x)*sum(y)-sum(x)*sum(x.*y))/(n*sum(x.*x)-sum(x)*sum(x));
y_hat = k*x+b; % 计算y的拟合值
y_2021 = k*2021+b;% 计算2021，y的预测值
SSR = sum((y_hat-mean(y)).^2);  % 回归平方和
SSE = sum((y_hat-y).^2); % 误差平方和
SST = sum((y-mean(y)).^2); % 总体平方和
SST-SSE-SSR;
R_2 = SSR / SST;% 拟合优度
plot_fit(x,y,k,b)% 画图
end
