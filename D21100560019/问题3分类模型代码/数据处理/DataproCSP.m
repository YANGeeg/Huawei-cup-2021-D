clc,clear;
%  TrainData=xlsread('Molecular_Descriptor',1,'B2:ABB1975');%读入数据
%  TextData=xlsread('Molecular_Descriptor',2,'B2:ABB51');%读入数据
%  Data=cat(1,TrainData,TextData);
% % 
TrainData=xlsread('data',1,'A2:T1975');%读入数据
TextData=xlsread('data_test',1,'A2:T51');%读入数据
Data=cat(1,TrainData,TextData);
TrueData=Data;
index=[];
% for i=1:length(Data(1,:))
%       if sum(Data(:,i)==0)<0.95*length(Data(:,i))
%           TrueData=[TrueData Data(:,i)];
%       end
%       if sum(Data(:,i)==0)>=0.95*length(Data(:,i))
%           index=[index i];
%       end
% end 
% for i=1:length(Data(1,:))
%       if sum(Data(:,i)==0)<0.95*length(Data(:,i))
%            TrueData=[TrueData Data(:,i)];
%        end
%        if sum(Data(:,i)==0)>=0.95*length(Data(:,i))
%            index=[index i];
%       end
% end 

num=length(TrueData(1,:));%剩余个数

Label=xlsread('ADMET',1,'B2:F1975');%读入数据

Ltext=zeros(50,1);
Ltext(:,:)=0;%假设标签

l1=cat(1,Label(:,1),Ltext);
l2=cat(1,Label(:,2),Ltext);
l3=cat(1,Label(:,3),Ltext);
l4=cat(1,Label(:,4),Ltext);
l5=cat(1,Label(:,5),Ltext);

% 
Data_allc(:,1,:)=permute(TrueData,[2,1]);
save TrueData.mat TrueData
save('Data_allc.mat','Data_allc')
% Data_allc0(:,1,:)=permute(Data,[2,1]);
% % save TrueData2.mat TrueData
% save('Data_allc0.mat','Data_allc0')

L_mc1=zeros(2024,3);
L_mc2=zeros(2024,3);
L_mc3=zeros(2024,3);
L_mc4=zeros(2024,3);
L_mc5=zeros(2024,3);
for i=1:3
    L_mc1(:,i)=l1;
    L_mc2(:,i)=l2;
    L_mc3(:,i)=l3;
    L_mc4(:,i)=l4;
    L_mc5(:,i)=l5;
end

save('l1.mat','l1')%
save('l2.mat','l2')%
save('l3.mat','l3')%
save('l4.mat','l4')%
save('l5.mat','l5')%

save('L_mc1.mat','L_mc1')%
save('L_mc2.mat','L_mc2')%
save('L_mc3.mat','L_mc3')%
save('L_mc4.mat','L_mc4')%
save('L_mc5.mat','L_mc5')%