clear all
clc
close all

dr = 'ProjectDiabeticRetinopathy.xlsx';
data = xlsread(dr,3);
%data(:,1:8)= zscore(data(:,1:8));

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

hiddenLayerSize = 15;
NoHiddenLayers = 2;

net = patternnet(hiddenLayerSize);
view(net)

[net,tr] = train(net,pred_train',output_train');
nnstart

% Test 

y = net(pred_test');
p = perform(net,output_test',y);
errors = gsubtract(output_test',y);
class = vec2ind(y);

classvsout = confusionmat(output_test',class);


