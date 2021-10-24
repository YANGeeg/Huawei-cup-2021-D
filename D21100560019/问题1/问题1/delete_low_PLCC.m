plcc = [];
for i=1:403
    index = IQAPerformance(Input_wo_zero(:,i), Output, 'p');
    plcc = [plcc;index];
end
count0 = 0;
count1 = 0;
count2 = 0;
count3 = 0;
count4 = 0;
count5 = 0;
count6 = 0;
for j=1:403
    if plcc(j,1) > 0 & plcc(j,1) < 0.1
        count0 = count0+1;
    end
    if plcc(j,1) > 0.1 & plcc(j,1) < 0.2
        count1 = count1+1;
    end
    if plcc(j,1) > 0.2 & plcc(j,1) < 0.3
        count2 = count2+1;
    end
    if plcc(j,1) > 0.3 & plcc(j,1) < 0.4
        count3 = count3+1;
    end
    if plcc(j,1) > 0.4 & plcc(j,1) < 0.5
        count4 = count4+1;
    end
    if plcc(j,1) > 0.5 & plcc(j,1) < 0.6
        count5 = count5+1;
    end
    if plcc(j,1) > 0.6 & plcc(j,1) < 0.7
        count6 = count6+1;
    end
end
Input_high_plc = [];
Factor_high_plc = cell(403,1);
tmp = 1
for k=1:403
    if plcc(k,1) > 0.3
        Input_high_plc = [Input_high_plc,Input_wo_zero(:,k)];
        Factor_high_plc{tmp,1} = Factor_wo_zero{k,1};
        tmp = tmp+1;
    end
end