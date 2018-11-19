clear all
clc
close all

bo = readtable('Building Occupancy.xlsx');
date = table2array(bo(:,2),'datetime');
time = table2array(bo(:,3),'datetime');
pmam = table2array(bo(:,4));
weekday = table2array(bo(:,5));


bodata = table2array(bo(:,6:10));
dataTIME = [time,weekday,bodata];
output = table2array(bo(:,11));
compositedata = [dataTIME output];

C= corrcoef(compositedata);

figure()
corrplot(C,'varNames',{'Time of Day', 'Day of Week','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy'})


C2=corrcoef(bodata(:,3),time);
C3 = corrcoef(bodata(:,3),weekday);

labels = {'Time of Day', 'Day of Week','Temperature','Humidity','Light','CO2','HumidityRatio'};

tTIME = fitctree(dataTIME,output,'PredictorNames',labels); % time of day as time lapsed

PredictorTime = nan(20560,7);
PredictorWeekday = nan(20560,7);
PredictorTemperature = nan(20560,7);
PredictorHumidity = nan(20560,7);
PredictorLight = nan(20560,7);
PredictorCO2 = nan(20560,7);
PredictorHumidityRatio = nan(20560,7);

predictors = {PredictorTime PredictorWeekday PredictorTemperature PredictorHumidity PredictorLight PredictorCO2 PredictorHumidityRatio};

%% Mean values for all columns 
baseMean = mean(dataTIME);
baseData = zeros(20560,7);
for i =1:size(baseData,2)
    for j = 1:size(baseData,1)
        baseData(j, i) = baseMean(i);
    end
end

for i = 1:size(baseData, 2)
    t1 = baseData;
    t1(:, i) = dataTIME(:,i);
    meanPreds{i} = t1;
end
%%
for i = 1:size(predictors,2)
    rnum = cell2mat(predictors(1,i));
    rnum(:,i) = dataTIME(:,i);
    predictors{i} = rnum;
end

for i =1:size(predictors,2)
    rnum = cell2mat(predictors(1,i));
    Newoutput{i} = predict(tTIME,rnum);
end


for j =1:size(meanPreds,2)
    rnum = cell2mat(meanPreds(1,j));
    NewoutputmeanPreds{j} = predict(tTIME,rnum);
end


UnifiedDataOutput = predict(tTIME,dataTIME);
OutPutError = confusionmat(output,UnifiedDataOutput);
UnifiedACC = trace(OutPutError)/sum(OutPutError(:))*100;

for i=1:size(NewoutputmeanPreds,2)
    ConfusionMatrixError{i} = confusionmat(output,NewoutputmeanPreds{1,i});
    ConfusionMatrixErrorNaN{i} = confusionmat(output,Newoutput{1,i});
end

for i = 1:size(ConfusionMatrixError,2)
    CM = cell2mat(ConfusionMatrixError(1,i));
    CMNaN = cell2mat(ConfusionMatrixErrorNaN(1,i));
    acc{i} = trace(CM)/sum(CM(:))*100;
    accNaN{i} = trace(CMNaN)/sum(CMNaN(:))*100;
end

for i =1:size(Newoutput,2)
    cnum = cell2mat(Newoutput(1,i));
    figure(1)
    subplot(2,4,i)
    scatter(dataTIME(:,i),cnum,'r.')
    ax=gca;
    ylim([0 1]);
    set(ax,'YTick',[0:0.5:1])
    title(['Constant Value was NaN - Occupancy VS' labels(i)]);
    xlabel(labels(i));
    ylabel('Predicted Occupany');
    grid
end

for i =1:size(NewoutputmeanPreds,2)
    cnum = cell2mat(NewoutputmeanPreds(1,i));
    figure(2)
    subplot(2,4,i)
    scatter(dataTIME(:,i),cnum,'r.')
    ax=gca;
    ylim([0 1]);
    set(ax,'YTick',[0:0.5:1])
    title(['Mean based - Occupancy VS' labels(i)]);
    xlabel(labels(i));
    ylabel('Predicted Occupany');
    grid
end

figure()
scatter3(weekday,time,Newoutput{5})
xlabel('Day of the Week (1: Sunday, 7: Saturday)')
xlim([1 7])
ylabel('Time of the Day')
zlabel('Predicted Occupany')
title('3D plot depicting occupancy as a function of Time of Day and Day of the Week')



view(tTIME,'Mode','graph')


Y1 = length(time(Newoutput{1} ==1));
X1 = length(weekday(Newoutput{2}==1));

Y2 = time(Newoutput{1} ==0);
X2 = weekday(Newoutput{2}==0);

figure()
scatter(X1,Y1,'k*')
hold on
scatter(X2,Y2,'r.')

imp = predictorImportance(tTIME);

figure()
bar(imp)
xlabel('Predictors');
ylabel('Importance')
title('Predictor Importance - CART model')


% RF


btime = TreeBagger(100,dataTIME,output,'InBagFraction',0.5,'Method','classification','NumPredictorsToSample',4);

bbtime = TreeBagger(100,dataTIME,output,'InBagFraction',0.5,'Method','classification','NumPredictorsToSample',4,...
    'OOBPredictorImportance','on','OOBPrediction','on');

figure()
bar(bbtime.DeltaCritDecisionSplit)
xlabel('Predictors');
ylabel('Importance')
title('Predictor Importance - RF model')

figure()
plot(oobError(bbtime))
xlabel('No. of Trees')
ylabel('OOB Error')
title('Out of Bag Error vs No. of Trees')

twomostimportant = baseData;
twomostimportant(:,3:2:5) = dataTIME(:,3:2:5);

twomostimportantNaN = nan(20560,7);
twomostimportantNaN(:,3:2:5) = dataTIME(:,3:2:5);

Light = dataTIME(:,5);
Temperature = dataTIME(:,3);

RFoutput = predict(btime,twomostimportant);
RFoutputNaN = predict(btime,twomostimportantNaN);
R = cell2mat(RFoutput(:,1));
RNaN = cell2mat(RFoutputNaN(:,1));

for i=1:length(R);
    RR(i) = str2double(R(i,1));
end

for i=1:length(RNaN);
    RRNaN(i) = str2double(RNaN(i,1));
end

RR = RR';
RRNaN = RRNaN';

OutPutErrorRF = confusionmat(output,RR);
OutPutErrorRFNaN = confusionmat(output,RRNaN);

AccRFMean = trace(OutPutErrorRF)/sum(OutPutErrorRF(:))*100;
AccRFNaN = trace(OutPutErrorRFNaN)/sum(OutPutErrorRFNaN(:))*100;


YY1 = Light(RR ==1);
XX1 = Temperature(RR==1);

YY2 = Light(RR ==0);
XX2 = Temperature(RR==0);

figure()
scatter(XX1,YY1,'k*')
hold on
scatter(XX2,YY2,'r.')
xlabel('Temperature - Second most Important variable')
ylabel('Light - The most important variable')
title('Predcited Ocupancy with the two most important variables');
legend('Occupied','Not Occupied')
grid

% Part 4

n = [1 3 5 6];

reducedata = dataTIME(:,n);

breduced = TreeBagger(100,reducedata,output,'InBagFraction',0.5,'Method','classification','NumPredictorsToSample',4,...
    'OOBPredictorImportance','on','OOBPrediction','on');


figure()
bar(breduced.DeltaCritDecisionSplit)
xlabel('Predictors');
ylabel('Importance')
title('Reduced Dataset Predictor Importance - RF model')

figure()
plot(oobError(breduced))

nnstart

