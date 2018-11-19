clear all
clc
close all

dr = 'ProjectDiabeticRetinopathy.xlsx';
data = xlsread(dr,4);

FullData = xlsread(dr,1);

C_FullData = corrcoef(FullData);

C = corrcoef(data);

newdata = data;

m = randperm(size(newdata,1));

newdatatrain = newdata(m(1:860),:);
newdatatest = newdata(m(861:1151),:);



pred_train = newdatatrain(:,1:(size(newdata,2)-1));
output_train = newdatatrain(:,size(newdata,2));

pred_test = newdatatest(:,1:(size(newdata,2)-1)); 
output_test = newdatatest(:,size(newdata,2));

% CART and RF model via 5 fold cross-validation on train dataset 


ind = crossvalind('Kfold',pred_train(:,1),5);

group1 = (ind==1); 
group2 = (ind==2); 
group3 = (ind==3); 
group4 = (ind==4); 
group5 = (ind==5); 
groups = {group5 group4 group3 group2 group1};

cv1 = any([group1,group2,group3,group4],2);
cv2 = any([group1,group2,group3,group5],2);
cv3 = any([group1,group2,group4,group5],2);
cv4 = any([group1,group3,group4,group5],2);
cv5 = any([group2,group3,group4,group5],2);

cv = {cv1 cv2 cv3 cv4 cv5};

%%


for i=1:length(cv)
    
    columnnumbercv = cell2mat(cv(1, i));  
    columnnumbergroup = cell2mat(groups(1,i));
    xtrain{i} = pred_train(columnnumbercv,:);
    xtest{i} = pred_train(columnnumbergroup,:);
    ytrain{i} = output_train(columnnumbercv,:);
    ytest{i} = output_train(columnnumbergroup,:);
    SVMLD = fitcsvm(cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)),'KernelFunction','linear','BoxConstraint',1,'ClassNames',[0,1]);
    %SVMLD = fitcsvm(cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)),'KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',inf,'ClassNames',[0,1]);
    [ResCV{i},scores{i}] = predict(SVMLD,cell2mat(xtest(1,i)));
    confusionMatCV = confusionmat(cell2mat(ytest(1,i)),cell2mat(ResCV(1,i)));
    accCV(i) = trace(confusionMatCV)/sum(confusionMatCV(:))*100;
end



SVM = fitcsvm(pred_train,output_train,'KernelFunction','linear','BoxConstraint',1,'CrossVal','on','ClassNames',[0,1]);

for j = 1:length(SVM.Trained)
    
    [I_SVM{j},scoresSVM] = predict(SVM.Trained{j},pred_test);
    CMTestSVM = confusionmat(output_test,cell2mat(I_SVM(1,j)));
    acc(j) = trace(CMTestSVM)/sum(CMTestSVM(:))*100;
end

[I,scoresI] = predict(SVMLD,pred_test);
CMTest = confusionmat(output_test,I);
accSVM = trace(CMTest)/sum(CMTest(:))*100;

False_Negatives = 100-((CMTest(2,1)/sum(CMTest(2,1:2)))*100)
False_Positives = 100-((CMTest(1,2)/sum(CMTest(1,1:2)))*100)
Reduction = ((abs((CMTest(1,2)+ CMTest(2,2))-sum(CMTest(:)))/sum(CMTest(:)))*100)  


d = 0.7;

[x1Grid,x2Grid] = meshgrid(min(pred_train(:,1)):d:max(pred_train(:,1)),min(pred_train(:,2)):d:max(pred_train(:,2)));

[~,S] = predict(SVMLD,[x1Grid(:),x2Grid(:),x1Grid(:),x2Grid(:),x1Grid(:),x2Grid(:)]);

figure()
h(1:2) = gscatter(pred_train(:,1),pred_train(:,2),output_train,'rb','.');
hold on
h(3) = plot(pred_train(SVMLD.IsSupportVector,1),pred_train(SVMLD.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(S(:,2),size(x1Grid)),[0 0],'k');
hold off
grid
xlabel('Hemorrahage at 50% Confidence')
ylabel('Hemorrahage at 60% Confidence')
title('Support Vector Machine Hyperplane')



