clear all
clc
close 

X = -10:1:10;
Y = -10:1:10;

for i = 1:size(X,2);
    for j = 1:size(Y,2);
        Z(i,j) = ((1-X(i))^2) + 100*((Y(j) - ((X(i))^2))^2);
    end
end

figure()
surf(X,Y,Z)

figure()
x = -5:0.1:5;
[X,Y] = meshgrid(x);
contour3(X,Y,Z,500)





