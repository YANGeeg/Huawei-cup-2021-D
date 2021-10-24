select_zero = [];
for i=1: 729
    count = 0;
    rate = 0;
    total = 1974;
    for j=1:1974
        if Input(j, i)==0
            count = count+1;
        end
    end
    rate = count / total;
    select_zero = [select_zero; rate];
end

Input_wo_zero = [];
Factor_wo_zero = cell(729,1);
tmp = 1
for k=1:729
    if select_zero(k, 1) < 0.95
        Input_wo_zero = [Input_wo_zero, Input(:,k)];
        Factor_wo_zero{tmp, 1} = Factors{k,1};
        tmp = tmp+1;
    end
end
