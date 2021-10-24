clc;clear;
load('heart_scale.mat') 
model = svmtrain(heart_scale_label,heart_scale_inst); 
[predict_label,accuracy] = svmpredict(heart_scale_label,heart_scale_inst,model);
[bestacc,bestc,bestg] = SVMcg(heart_scale_label,heart_scale_inst,-9,9,-3,3,10,1,1);
