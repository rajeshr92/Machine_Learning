clear all
clc
close all

load('VarName1.mat')
load('VarName2.mat')


a = VarName1;
b = VarName2;
Var = [a,b];

% 
% C = corrcoef(a,b);
% 
% id = kmeans(Var,2);
% %[sil,h] = silhouette(Var,id);
% %dist = mean(sil);
% 
% 
% [id,h] = kmeans(Var,30);
% 
% figure()
% scatter(a,b,1,id)
% hold on
% scatter(h(:,1),h(:,2),'kx','LineWidth',3)
% hold off

%[sil30,h30] = silhouette(Var,id);


for i = 2:90
    i
    [idl,C,D] = kmeans(Var,i);
    distD(i) = mean(D);
end

figure()
plot(1:58,distD)
xlabel('Clusters')
ylabel('Distance from centroid of each cluster')
grid
