clear all
clc
close all

%%% 1

AQMat = 'Air Quality Data Set.xlsx';
AQdata = xlsread(AQMat,1);
labelO = {'Date' 'Time' 'Day'	'(CO)'	'(NMHC)'	'(NOx)'	'(NO2)'	'(O3)'	'T'	'RH'	'AH'	'C6H6(GT)'};
labels = {'Date' 'Time' 'Day'	'(CO)'	'(NMHC)'	'(NOx)'	'(NO2)'	'(O3)'	'T'	'RH'	'AH'};

AQPred = AQdata(:,1:11);
output = AQdata(:,12);

AQPredMin = min(zscore(AQPred));
AQPredMax = max(zscore(AQPred));

% AQdata = readtable('Air Quality Data Set.xlsx');
% 
% AQ = AQdata(:,1:11);
% date = table2array(AQ(:,1),'datetime');
% datee = cell2double(date);

%%% 2

C = corrcoef(AQdata);
corrplot(C,'varNames',labelO)

% linear regression model

normPred = zscore(AQPred);
normOutput = zscore(output);

mdl = fitlm(normPred,normOutput);
pvalue = mdl.Coefficients(:,4);
fitted = mdl.Fitted;

figure()
plot(mdl)
grid on

tree = fitrtree(normPred,normOutput,'PredictorNames',labels);
ttree = fitrtree(AQPred,output,'PredictorNames',labels);

imp = predictorImportance(tree);
imptt = predictorImportance(ttree);

figure()
bar(imp)
xlabel('Predictors');
ylabel('Importance')
title('Predictor Importance - Regression Tree')
grid

figure()
bar(imptt)
xlabel('Predictors');
ylabel('Importance')
title('Predictor Importance - Regression Tree')
grid


%[beta,SE,inmodel,stats,nextstep,history] = stepwisefit(AQPred,output);
% Ymodel = beta(1)*AQPred(:,1) + beta(2)*AQPred(:,2) + beta(3)*AQPred(:,3)...
%     + beta(4)*AQPred(:,4) + beta(5)*AQPred(:,5)+ beta(6)*AQPred(:,6)+...
%     beta(7)*AQPred(:,7) + beta(8)*AQPred(:,8) + beta(9)*AQPred(:,9) +...
%     beta(10)*AQPred(:,10)+ nextstep.intercept;


for i = 1:length(fitted)
    error(i) = ((fitted(i)-normOutput(i))^2);
end

SSE = sum(error);
TSS = sum((normOutput - mean(normOutput)).^2);
Rlm = 1- (SSE/TSS);


% ANN

x = zscore(AQPred');
t = zscore(output');

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = 15;
NoHiddenLayers = 2;

%net = feedforwardnet([15,15]); % default 
net = fitnet(hiddenLayerSize,trainFcn)
%net.layers{1}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
net.layers{1}.transferFcn = 'logsig';



[net,tr] = train(net,x,t);   % Train the Network





% Just using either predictor 3, 5 or 7 in the net
XMin = zeros(9357,11);
XMax = zeros(9357,11);
X3 = zeros(9357,11);
X5 = zeros(9357,11);
X7 = zeros(9357,11);
predmean = mean(normPred);
imppreds = zeros(size(AQPred));


for i =1:size(XMin,2)
    for j = 1:size(XMin,1)
        XMin(j, i) = AQPredMin(i);
        XMax(j,i) = AQPredMax(i);
        imppreds(j, i) = predmean(i);
    end
end


%cellX = {XMin, XMin, XMin};
%cellX = {XMax, XMax, XMax};
cellX = {imppreds, imppreds, imppreds};
j=3;

for i = 1:size(cellX,2)
    if j<=7
    rnum = cell2mat(cellX(1,i));
    rnum(:,j) = zscore(AQPred(:,j));
    cellX{i} = rnum;
    j = j+2;
    end
end


y = net(x);

for i = 1:size(cellX,2)
    cnum = cell2mat(cellX(1,i))';
    y357{i} = net(cnum);
end

NewX3 = cell2mat(cellX(1,1))';
NewX5 = cell2mat(cellX(1,2))';
NewX7 = cell2mat(cellX(1,3))';

y1 = net(NewX7);

e = gsubtract(t,y);
performance = perform(net,t,y)  % Test the Network

% View the Network
view(net)


figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotregression(t,cell2mat(y357(1,3)))
figure, plotfit(net,NewX3(3,:),cell2mat(y357(1,1)))
figure, plotfit(net,NewX5(5,:),cell2mat(y357(1,2)))
figure, plotfit(net,NewX7(7,:),cell2mat(y357(1,3)))
figure, plotfit(net,x(5,:),y)

figure, scatter(NewX3(3,:),cell2mat(y357(1,1)))
figure, scatter(x(5,:),y)


figure,scatter(NewX3(3,:)',cell2mat(y357(1,1))')
figure,scatter(NewX3(5,:)',cell2mat(y357(1,2))')
figure,scatter(NewX3(7,:)',cell2mat(y357(1,3))')

%err = mse(trained,inputs,outputs,'regularization',0.01);

%%% Picking the top two predictors while keeping rest at mean value


imppreds(:,5:2:7) = zscore(AQPred(:,5:2:7));


xx = imppreds';

yy = net(xx);
ee = gsubtract(t,y);
performanceyy = perform(net,t,yy)  % Test the Network

figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(ee)
figure, plotregression(t,yy)
figure, plotfit(net,xx(5,:),t)

figure()
scatter(imppreds(:,5),yy)
hold on
scatter(imppreds(:,5),t)
xlabel('Titania Concentration')
ylabel('ANN Predicted Output')
title('Response plot - Titania vs. Benzene Concentration')
legend('Predicted with 10 neurons' , 'Actual Outcome')
grid

figure()
scatter(imppreds(:,7),yy)
xlabel('Tungsten Concentration')
ylabel('ANN Predicted Output')
title('Response plot - Tungsten Oxide vs. Benzene Concentration')
grid

% sense

x7 = linspace(min(imppreds(:,7)), max(imppreds(:,7)), 9357);
x5 = linspace(min(imppreds(:,5)), max(imppreds(:,5)), 9357);

imppreds(:,7) = x7;
xsense = imppreds';


ysense = net(xsense);

figure()
scatter(imppreds(:,5),ysense)
hold on
scatter(imppreds(:,5),yy,'.')
xlabel('Titania Concentration')
ylabel('ANN Predicted Output')
title('Sensitivity plot - Titania vs. Benzene Concentration')
legend('Tungsten Oxide from min to max', 'Titania and Tungsten Oxide left as is')
grid

%%% Tweaking 

netFF = feedforwardnet([15,15]);
netcnet = layrecnet(1,[15,15]);

view(netFF)
view(netcnet)





