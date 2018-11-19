clear all
clc
close all

dr = 'ProjectDiabeticRetinopathy.xlsx';
data = xlsread(dr,1);

%data(:,1:19) = zscore(data(:,1:19));

C = corrcoef(data);
corrplot(data)
% Train dataset 

m = randperm(size(data,1));
dataTrain = data(m(1:860),:);
pred_train = dataTrain(:,1:19);
output_train = dataTrain(:,20);

% Test dataset

dataTest = data(m(861:1151),:);
pred_test = dataTest(:,1:19);
output_test = dataTest(:,20);


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
    
    % CART Model
    
    t = fitctree(cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)),'PredictorNames'...
        ,{'Quality', 'Pre-screening','MA0.5','MA0.6','MA0.7','MA0.8','MA0.9',...
        'MA1','EXU1','EXU2','EXU3','EXU4','EXU5','EXU6','EXU7'...
        'EXU8','MacDist','OpticDisc','AMFM'});
    
%     imp = predictorImportance(t);
%     figure()
%     bar(imp);
    
    
    [I{i},~] = predict(t,cell2mat(xtest(1,i)));
    confusionMat = confusionmat(cell2mat(ytest(1,i)),cell2mat(I(1,i)));
    acc(i) = ((trace(confusionMat))/(sum(confusionMat(:))))*100;
    
    % Random Forest Model
    
    RF = TreeBagger(1000,cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)),...
        'InBagFraction',0.4,'Method','classification','NumPredictorsToSample',8,...
        'OOBPredictorImportance','on','OOBPrediction','on');
    
%     figure()
%     bar(RF.DeltaCritDecisionSplit)
    

    
    [IRandFor{i},~] = predict(RF,cell2mat(xtest(1,i)));
    confusionMatRF = confusionmat(cell2mat(ytest(1,i)),str2num(cell2mat((IRandFor{1,i}))));

     accRF(i) = ((trace(confusionMatRF))/(sum(confusionMatRF(:))))*100;

end




figure()
plot(oobError(RF))
xlabel('Number of Trees');
ylabel('Importance')
title('Out of Bag error - RF model')
grid 

figure()
bar(RF.DeltaCritDecisionSplit)
xlabel('Predictors');
ylabel('Importance')
title('Predictor Importance - RF model')

mean(acc)
mean(accRF)

[PredictedTestOutput,~] = predict(RF,pred_test);
confusionMatTest = confusionmat(output_test,str2num(cell2mat(PredictedTestOutput)));
accTEST = trace(confusionMatTest)/sum(confusionMatTest(:))*100;

False_Negatives = ((confusionMatTest(2,1)/sum(confusionMatTest(2,1:2)))*100)
False_Positives = ((confusionMatTest(1,2)/sum(confusionMatTest(1,1:2)))*100)
Reduction = ((abs((confusionMatTest(1,2)+ confusionMatTest(2,2))-sum(confusionMatTest(:)))/sum(confusionMatTest(:)))*100)  



% Feature Selection 3,9,17, and 18






%corrplot(data,'varNames',{'Quality', 'Pre-screening','MA0.5','MA0.6','MA0.7','MA0.8','MA0.9','MA1','EXU1'	'EXU2'	'EXU3'	'EXU4'	'EXU5'	'EXU6'	'EXU7'	'EXU8'	'MacDist'	'OpticDisc'	'AMFM'	'Outcome' })
%corrplot(data)
%corrplot(data,'varNames',{'1', '2','3','4','5','6','7','8','9'	'10' '11' '12'	'13'	'14'	'15'	'16'	'17'	'18'	'19'	'20' })


