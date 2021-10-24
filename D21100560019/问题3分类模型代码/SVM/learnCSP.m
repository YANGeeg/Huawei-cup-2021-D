function CSPMatrix = learnCSP(EEGSignals,classLabels)
%
% 输入：
% EEGSignals：训练EEG信号，由2个类组成。 这些信号
% 结构是这样的：
% EEGSignals.x：EEG信号以[Ns * Nc * Nt]矩阵表示，其中
% Ns：每次试验的脑电图样本数量
% Nc：通道数（EEG电极）
% nT：试验次数
% EEGSignals.y：[1 * Nt]向量，包含每个试验的类标签
% EEGSignals.s：采样频率（以Hz为单位）

% 输出：
% CSPMatrix：学习的CSP过滤器（[Nc * Nc]矩阵，过滤器为行）
%
% 另请参见：extractCSPFeatures

%检查和初始化
nbChannels = size(EEGSignals.x,2);
nbTrials = size(EEGSignals.x,3);
nbClasses = length(classLabels);

if nbClasses ~= 2
    disp('ERROR! CSP can only be used for two classes');
    return;
end

covMatrices = cell(nbClasses,1); %the covariance matrices for each class

%% Computing the normalized covariance matrices for each trial
trialCov = zeros(nbChannels,nbChannels,nbTrials);
for t=1:nbTrials
    E = EEGSignals.x(:,:,t)';                       %note the transpose
    EE = E * E';
    trialCov(:,:,t) = EE ./ trace(EE);
end
clear E;
clear EE;

%computing the covariance matrix for each class
for c=1:nbClasses      
    covMatrices{c} = mean(trialCov(:,:,EEGSignals.y == classLabels(c)),3); %EEGSignals.y==classLabels(c) returns the indeces corresponding to the class labels  
end

%the total covariance matrix
covTotal = covMatrices{1} + covMatrices{2};

%whitening transform of total covariance matrix
[Ut Dt] = eig(covTotal); %caution: the eigenvalues are initially in increasing order
eigenvalues = diag(Dt);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
Ut = Ut(:,egIndex);
P = diag(sqrt(1./eigenvalues)) * Ut';

%transforming covariance matrix of first class using P
transformedCov1 =  P * covMatrices{1} * P';

%EVD of the transformed covariance matrix
[U1 D1] = eig(transformedCov1);
eigenvalues = diag(D1);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
U1 = U1(:, egIndex);
CSPMatrix = U1' * P; 