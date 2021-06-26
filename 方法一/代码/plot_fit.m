function []=plot_fit(x,y,k,b)
% 画出样本散点图
plot(x,y,'o')
% 给x和y轴加上标签
xlabel('x的值')
ylabel('y的值')
hold on % 继续在之前的图形上来画图形
grid on % 显示网格线
% % 画出y=kx+b的函数图像 plot(x,y)
% % 传统的画法：模拟生成x和y的序列，比如要画出[0,5]上的图形
% x = 0: 0.1 :5  % 间隔设置的越小画出来的图形越准确
% y = k * x + b  % k和b都是已知值
% plot(x,y,'-')
f=@(x) k*x+b;
fplot(f,[min(x)-1,max(x)+1]);
legend('样本数据','拟合函数','location','SouthEast')
end
% 匿名函数的基本用法。
% handle = @(arglist) anonymous_function
% 其中handle为调用匿名函数时使用的名字。
% arglist为匿名函数的输入参数，可以是一个，也可以是多个，用逗号分隔。
% anonymous_function为匿名函数的表达式。
% 举个小例子
% % >> z=@(x,y) x^2+y^2; 
% % >> z(1,2) 
% % ans =  5
% fplot函数可用于画出匿名函数的图形。
% fplot(f,xinterval) 将在指定区间绘图。将区间指定为 [xmin xmax] 形式的二元素向量。