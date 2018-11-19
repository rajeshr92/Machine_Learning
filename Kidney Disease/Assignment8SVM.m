clear all
clc
close all

data = 'ckdclass.xlsx';
datatable = readtable('ckdclass.xlsx');
blah = xlsread(data);

% data processing 

x = table2cell(datatable(:, 6:7));
x(strcmp('normal', x)) = {0};
x(strcmp('abnormal', x)) = {1};
x(strcmp('?', x)) = {NaN};

x = cell2mat(x);

y = table2cell(datatable(:,8:9));
y(strcmp('notpresent',y)) = {0};
y(strcmp('present',y)) = {1};
y(strcmp('?',y)) = {NaN};

y = cell2mat(y);

z = table2cell(datatable(:,19:23));
z(strcmp('no',z)) = {0};
z(strcmp('yes',z)) = {1};
z(strcmp('',z)) = {NaN};
z(strcmp('?',z)) = {NaN};
z(strcmp('good',z)) = {0};

z = cell2mat(z);

w = table2cell(datatable(:,24));
w(strcmp('',w)) = {NaN};
w(strcmp('?',w)) = {NaN};
w(strcmp('good',w)) = {0};
w(strcmp('no',w)) = {1};
w(strcmp('poor',w)) = {1};

w = cell2mat(w);

v = table2cell(datatable(:,25));
v(strcmp('',v)) = {NaN};
v(strcmp('ckd',v)) = {1};
v(strcmp('notckd',v)) = {0};
v(strcmp('no',v)) = {0};

v = cell2mat(v);

newdata =[blah(:,1:5),x(:,1),x(:,2),y(:,1),y(:,2),blah(:,10:18),z(:,1:5),w(:,1),v(:,1)];

ListDel = newdata;
ListDel(any(isnan(ListDel),2),:) = [];

datamean = nanmean(newdata);
D = isnan(newdata);
MeanDriven = newdata;

for j=1:size(D,2)
    if j<=25
    for i=1:size(D,1)
        if D(i,j)== 1 
            MeanDriven(i,j) = datamean(j);
        end
    end
    end
end

% Normalizing ListDel and MeanDriven

a = zscore(ListDel);
b = zscore(MeanDriven);

apred = a(:,1:24);
aout = ListDel(:,25);
C = corrcoef(ListDel);
%corrplot(C)

bpred = b(:,1:24);
bout = newdata(:,25);

% SVM ListDel (LD)

svmplotLD = svmtrain(apred,aout);
[Res] = svmclassify(svmplotLD,apred);
confusionMat = confusionmat(aout,Res);
acc = trace(confusionMat)/sum(confusionMat(:))*100


ind = crossvalind('Kfold',apred(:,1),5);

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
    xtrain{i} = apred(columnnumbercv,:);
    xtest{i} = apred(columnnumbergroup,:);
    ytrain{i} = aout(columnnumbercv,:);
    ytest{i} = aout(columnnumbergroup,:);
    %SVMLD{i} = fitcsvm(cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)),'KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',inf,'ClassNames',[0,1]);
    SVMLD{i} = fitcsvm(cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)),'KernelFunction','polynomial','BoxConstraint',inf,'ClassNames',[0,1]);
    SVMModel{i} = svmtrain(cell2mat(xtrain(1,i)),cell2mat(ytrain(1,i)));
    SVMModelClass{i} = svmclassify(SVMModel{1,i},cell2mat(xtest(1,i)));
    [ResCV{i},scores{i}] = predict(SVMLD{1,i},cell2mat(xtest(1,i)));
    confusionMatCV = confusionmat(cell2mat(ytest(1,i)),cell2mat(ResCV(1,i)));
    accCV(i) = trace(confusionMatCV)/sum(confusionMatCV(:))*100;
end





s = 4;

y1 = ytrain{1,s};
R1 = ResCV{1,s};
x1 = xtrain{1,s};
y2 = ytest{1,s};
x2 = xtest{1,s};
scoL = scores{1,s};

d = 1.1699;
f = 0.6027;


%[x1Grid,x2Grid] = meshgrid(min(x2(:,17)):d:max(x2(:,17)),min(x2(:,18)):f:max(x2(:,18)));

[x1Grid,x2Grid] = meshgrid(linspace(min(x2(:,17)),max(x2(:,17)),5),linspace(min(x2(:,18)),max(x2(:,18)),5));

figure()
gscatter(x1(:,17),x1(:,18),y1,'br','xo')
grid
xlabel('White Blood Cell count')
ylabel('Reb Blood Cell count')
title('Scatter plot of data points')


figure()
h(1:2) = gscatter(x1(:,17),x1(:,18),y1,'rb','.');
hold on
h(3) = plot(x1(SVMLD{1,s}.IsSupportVector,17),x1(SVMLD{1,s}.IsSupportVector,18),'ko');
contour(x1Grid,x2Grid,reshape(scoL(:,2),size(x1Grid)),[0 0],'k');
hold off
grid
xlabel('White Blood Cell count')
ylabel('Reb Blood Cell count')
title('Trained SVM model with Hyperplane')


figure()
gscatter(x2(:,17),x2(:,18),R1,'rb','.');
hold on
gscatter(x2(:,17),x2(:,18),y2,'gr','x*');
contour(x1Grid,x2Grid,reshape(scoL(:,2),size(x1Grid)),[0 0],'k');
hold off
xlabel('White Blood Cell count')
ylabel('Reb Blood Cell count')
title('Trained and Test result using SVM model with Hyperplane')


