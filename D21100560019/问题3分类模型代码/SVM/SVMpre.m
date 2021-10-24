%% 使用libsvm 运动想象分类
clc;
clear all;
%run_FeatureExtract;
disp('#######  Training The SVM Classsifier ##########')
load dataCSP1.mat 
load l1.mat
%% XCSP为训练集， TCSP测试集， YCSP训练标签
%% SVM网络训练
% model = svmtrain(l3(1:1580,:) , dataCSP1(1:1580,:), '-c 2 -g 1');
% %% SVM网络预测
% [predict_label, accuracy,decision_values] = svmpredict(l3(1581:1974,:), dataCSP1(1581:1974,:), model);

model = svmtrain(l1(1:1974,:) , dataCSP1(1:1974,:), '-c 2 -g 1');
[predict_label, accuracy,decision_values] = svmpredict(l1(1975:end,:), dataCSP1(1975:end,:), model);

% -s svm类型：SVM设置类型(默认0)
% 0 — C-SVC
% 1 –v-SVC
% 2 – 一类SVM
% 3 — e -SVR
% 4 — v-SVR
% 
% -t 核函数类型：核函数设置类型(默认2)
% 0 – 线性：u’v
% 1 – 多项式：(r*u’v + coef0)^degree
% 2 – RBF函数：exp(-r|u-v|^2)
% 3 –sigmoid：tanh(r*u’v + coef0)
% 
% -g r(gama)：核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)
% 
% -c cost：设置C-SVC，e -SVR和v-SVR的参数(损失函数)(默认1)，惩罚系数
% %% Visualization
% testset = dataCSP1(1581:1974,:);
% labels = label1(1581:1974,:);
% no_dims = 2;
% initial_dims = 2;
% perplexity = 30;
% % t-sne dimensionality reduction
% ans_map = tsne(testset, predict_label, no_dims, initial_dims, perplexity);
% gscatter(ans_map(:,1), ans_map(:,2), labels);
% title('TestSet Classification by SVM Visualization');