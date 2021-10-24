%% I. ��ջ�������
clear all
clc
%% II. ��������
% load dataCSP1.mat 
load TrueData.mat
load l1.mat
%%
% 2. ѵ�����D�D80������
% train_matrix = dataCSP1(1:1974,:);
train_matrix = dataCSP1(1:1974,:);
train_label = label1(1:1974,:);
 
%%
% 3. ���Լ��D�D26������
test_matrix = dataCSP1(1975:end,:);
test_label = label1(1975:end,:);
 
%% III. ���ݹ�һ��
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',test_matrix',PS);
Test_matrix = Test_matrix';
%% IV. SVM����/ѵ��(RBF�˺���)
%%
% 1. Ѱ�����c/g�����D�D������֤����
[c,g] = meshgrid(-5:0.2:5,-5:0.2:5);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
        cg(i,j) = svmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
 
%%
% 2. ����/ѵ��SVMģ��
model = svmtrain(train_label,Train_matrix,cmd);
 
%% V. SVM�������
[predict_label_1,accuracy_1,decision_values1] = svmpredict(train_label,Train_matrix,model); 
[predict_label_2,accuracy_2,decision_values2] = svmpredict(test_label,Test_matrix,model);

result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%% VI. ��ͼ
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('��ʵ���','Ԥ�����')
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�SVMԤ�����Ա�(RBF�˺���)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)
