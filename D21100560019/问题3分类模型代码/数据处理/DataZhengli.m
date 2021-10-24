clc,clear;
TrainData=xlsread('Molecular_Descriptor',1,'B2:ABB1975');%读入数据
TextData=xlsread('Molecular_Descriptor',2,'B2:ABB51');%读入数据

Data=cat(1,TrainData,TextData);
TrueData=[];%真正留下来的
index=[];%删除的数据索引
 for i=1:length(Data(1,:))
      TrueData=[TrueData Data(:,i)];
 %        if sum(Data(:,i)==0)<0.95*length(Data(:,i))
 %            TrueData=[TrueData Data(:,i)];
 %        end
 %        if sum(Data(:,i)==0)>=0.95*length(Data(:,i))
 %            index=[index i];
 %        end
 end 
Label=xlsread('ADMET',1,'B2:F1975');%读入数据
l1=Label(:,1);
l2=Label(:,2);
l3=Label(:,3);
l4=Label(:,4);
l5=Label(:,5);

Ltext=zeros(50,1);
Ltext(:,:)=0;%第3类

num=length(TrueData(1,:));%剩余个数

Data_all3(:,1,:)=TrueData; %%%

L_model1=cat(1,l1,Ltext);
L_model2=cat(1,l2,Ltext);
L_model3=cat(1,l3,Ltext);
L_model4=cat(1,l4,Ltext);
L_model5=cat(1,l5,Ltext);

L_m1=zeros(2024,num);
L_m2=zeros(2024,num);
L_m3=zeros(2024,num);
L_m4=zeros(2024,num);
L_m5=zeros(2024,num);
for i=1:num
    L_m1(:,i)=L_model1;
    L_m2(:,i)=L_model2;
    L_m3(:,i)=L_model3;
    L_m4(:,i)=L_model4;
    L_m5(:,i)=L_model5;
end

save('Data_all3.mat','Data_all3')%
save('L_729m1.mat','L_m1')%
save('L_729m2.mat','L_m2')%
save('L_729m3.mat','L_m3')%
save('L_729m4.mat','L_m4')%
save('L_729m5.mat','L_m5')%