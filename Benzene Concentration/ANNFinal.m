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


for i = 1:length(fitted)
    error(i) = ((fitted(i)-normOutput(i))^2);
end

SSE = sum(error);
TSS = sum((normOutput - mean(normOutput)).^2);
Rlm = 1- (SSE/TSS);


%% ANN

% ANN

x = zscore(AQPred');
t = zscore(output');

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = 15;
NoHiddenLayers = 2;

net = fitnet(hiddenLayerSize,trainFcn); % default 
%net.layers{1}.transferFcn = 'purelin';
net.layers{1}.transferFcn = 'tansig';
%net.layers{1}.transferFcn = 'logsig';



[net,tr] = train(net,x,t);   % Train the Network
y = net(x);

%%% Picking the top two predictors while keeping rest at mean value


predmean = mean(normPred);
imppreds = zeros(size(AQPred));


for i =1:size(imppreds,2)
    for j = 1:size(imppreds,1)
        imppreds(j, i) = predmean(i);
    end
end

imppreds(:,5:2:7) = zscore(AQPred(:,5:2:7));


xx = imppreds';

yy = net(xx);
ee = gsubtract(t,y);
performanceyy = perform(net,t,yy)  % Test the Network


figure()
scatter(imppreds(:,5),yy)
xlabel('Titania Concentration')
ylabel('ANN Predicted Output')
title('Response plot - Titania vs. Benzene Concentration')
grid

figure()
scatter(imppreds(:,7),yy)
xlabel('Tungsten Concentration')
ylabel('ANN Predicted Output')
title('Response plot - Tungsten Oxide vs. Benzene Concentration')
grid

% Sensitivity 

x7 = linspace(min(imppreds(:,7)), max(imppreds(:,7)), 9357);
imppreds(:,7) = x7;

newX = imppreds';

ynew = net(newX);


figure()
scatter(imppreds(:,5),ynew)
xlabel('Titania Concentration')
ylabel('ANN Predicted Output')
title('Sensitivity plot: Response plot - Titania vs. Benzene Concentration')
grid


