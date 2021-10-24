function [ best_performance2 , best_model , best_C , best_gamma, print ] = CL_libSVM( train , trainlabels , test , testlabels , c , g , folds , limit)
%CL_LIBSVM SVM Classifier with LIBSVM
%Usage:
%   [best_performance,best_model,best_C,best_gamma,print] = CL_libSVM(train,trainlabels,test,testlabels,c,g,folds,limit)
%Input:
%   train (events x features)
%   trainlabels (events x 1)
%   test (events x features)
%   testlabels (events x 1)
%   C (wanted log_2(C) values x 1)
%   g (wanted log_2(Gamma) values x 1)
%   folds (number of folds for cross-validation)
%   limit (number of events to process)
%Output:
%   best_performance (higher test accuracy value)
%   best_model (SVM model that resulted in best_performance)
%   best_C (C value of best_model)
%   best_gamma (Gamma value of best_model)
%   print (string for interface text feedback)

disp('------ SVM Classifier ------');

if limit==0
    limit = size(train,1);
    limittest = size(test,1);
else
    limittest = limit;
    if limittest>size(test,1)
        limittest = size(test,1);
    end
end

data = train(1:limit,:);
labels = trainlabels(1:limit);
datatest = test(1:limittest,:);
labelstest = testlabels(1:limittest);

[C,gamma] = meshgrid(c,g);

tic
%---Grid search and cross-validation
cv_acc = zeros(numel(C),1);
parfor i=1:numel(C)
    fprintf('Run %d/%d: ',i,numel(C));
    cv_acc(i) = libsvmtrain(labels, data, ...
                    sprintf('-c %f -g %f -v %d -q', 2^C(i), 2^gamma(i), folds));
end
toc

%---Pair (C,gamma) with best accuracy
[best_performance,idx] = max(cv_acc);
fprintf('Cross Validation maximum Accuracy = %f%% \n',best_performance);

%---Contour plot
figure()
contourf(C, gamma, reshape(cv_acc,size(C))), colorbar,
hold on
plot(C(idx), gamma(idx), 'rx')
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

%---Best parameters
best_C = 2^C(idx);
best_gamma = 2^gamma(idx);

fprintf('Best C = %f \n',best_C);
fprintf('Best Gamma = %f \n',best_gamma);

%---Train with best parameters
best_model = libsvmtrain(labels, data, ...
                    sprintf('-c %f -g %f -q', best_C, best_gamma));

%---Predict / Test
[~, accuracy,~] = libsvmpredict(labelstest(1:limittest),datatest(1:limittest,:), best_model, '-q');
best_performance2 = accuracy(1);

fprintf('Test Accuracy = %f%% \n',best_performance2);
disp('------------------------------');

print = sprintf('------ SVM Classifier ------ \nCross Validation maximum Accuracy = %f%% \nBest C = %f \nBest Gamma = %f \nTest Accuracy = %f%% \n------------------------------',best_performance,best_C,best_gamma,best_performance2);
end