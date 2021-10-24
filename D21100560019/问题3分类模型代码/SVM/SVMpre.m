%% ʹ��libsvm �˶��������
clc;
clear all;
%run_FeatureExtract;
disp('#######  Training The SVM Classsifier ##########')
load dataCSP1.mat 
load l1.mat
%% XCSPΪѵ������ TCSP���Լ��� YCSPѵ����ǩ
%% SVM����ѵ��
% model = svmtrain(l3(1:1580,:) , dataCSP1(1:1580,:), '-c 2 -g 1');
% %% SVM����Ԥ��
% [predict_label, accuracy,decision_values] = svmpredict(l3(1581:1974,:), dataCSP1(1581:1974,:), model);

model = svmtrain(l1(1:1974,:) , dataCSP1(1:1974,:), '-c 2 -g 1');
[predict_label, accuracy,decision_values] = svmpredict(l1(1975:end,:), dataCSP1(1975:end,:), model);

% -s svm���ͣ�SVM��������(Ĭ��0)
% 0 �� C-SVC
% 1 �Cv-SVC
% 2 �C һ��SVM
% 3 �� e -SVR
% 4 �� v-SVR
% 
% -t �˺������ͣ��˺�����������(Ĭ��2)
% 0 �C ���ԣ�u��v
% 1 �C ����ʽ��(r*u��v + coef0)^degree
% 2 �C RBF������exp(-r|u-v|^2)
% 3 �Csigmoid��tanh(r*u��v + coef0)
% 
% -g r(gama)���˺����е�gamma��������(��Զ���ʽ/rbf/sigmoid�˺���)
% 
% -c cost������C-SVC��e -SVR��v-SVR�Ĳ���(��ʧ����)(Ĭ��1)���ͷ�ϵ��
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