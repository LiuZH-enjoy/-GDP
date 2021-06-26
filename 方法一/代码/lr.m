function [y_2021,R_2] = lr(x,y)
% ����������
n = size(x,1);
% ������С���˷�����ϵ��
k = (n*sum(x.*y)-sum(x)*sum(y))/(n*sum(x.*x)-sum(x)*sum(x));
% ������С���˷�����ƫ����
b = (sum(x.*x)*sum(y)-sum(x)*sum(x.*y))/(n*sum(x.*x)-sum(x)*sum(x));
y_hat = k*x+b; % ����y�����ֵ
y_2021 = k*2021+b;% ����2021��y��Ԥ��ֵ
SSR = sum((y_hat-mean(y)).^2);  % �ع�ƽ����
SSE = sum((y_hat-y).^2); % ���ƽ����
SST = sum((y-mean(y)).^2); % ����ƽ����
SST-SSE-SSR;
R_2 = SSR / SST;% ����Ŷ�
plot_fit(x,y,k,b)% ��ͼ
end
