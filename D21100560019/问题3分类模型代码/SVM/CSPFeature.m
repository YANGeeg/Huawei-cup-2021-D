close all; clear; clc;

load Data_allc.mat  %Data_allc
load l1.mat

EEGSignals.x=Data_allc;%不标准化zscore
EEGSignals.y=l1;
label1=l1;


classLabels = unique(EEGSignals.y); 
CSPMatrix = learnCSP(EEGSignals,classLabels);
nbFilterPairs = 1;

dataCSP1 = extractCSP(EEGSignals, CSPMatrix, nbFilterPairs);  

save dataCSP1.mat dataCSP1 label1 
X=dataCSP1 ;
Y=label1;

color_L = [0 102 255] ./ 255;
color_R = [255, 0, 102] ./ 255;

pos = find(Y==1);
plot(X(pos,1),X(pos,2),'o','Color',color_L,'LineWidth',2);

hold on
pos1 = find(Y==0);
plot(X(pos1,1),X(pos1,2),'.','Color',color_R,'LineWidth',2);

legend('1','0')
xlabel('C3','fontweight','bold')
ylabel('C4','fontweight','bold')