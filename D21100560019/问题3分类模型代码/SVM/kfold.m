clc;
clear all;
%run_FeatureExtract;

load dataCSP2.mat 
for t=1:280
    if YCSP(t)==2;
      YCSP(t)=-1;
    end
end
% acc = svmtrain(YCSP, XCSP, '-t 2 -c 2 -g 1 -v 10 ')
stats = nFold_SVM(XCSP,YCSP,1,10,'rbf',10)