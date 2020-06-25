%The main help is taken from Andrew Ng's Machine Learning Course, I am a student of that course.
clear all;
close all;
clc;
data=load('PimaIndians.txt');
X = data(:, [1:8]); y = data(:, 9);
fprintf('Plotting Pima dataset: X would be the inputs and y would be output.\n');

plotData(X,y);
hold on;
xlabel('Xs')
ylabel('y')
legend('posz','Negz')
hold off;
%Now setup to calculate cost and gradient function
%adding ones to the start of X just to make it cleaner
[m,n]=size(X);
X=[ones(m,1) X];
%now X has 1 in its first column
%now we will initialize theta by deafult to 0
initial_theta=zeros(n+1, 1);
%how initial_theta is set??
%we take the length of columns in X and set it 0 and save it in
%inital_theta
%Now setup Cost Function
[cost,grad]=costFunction(initial_theta,X,y);
fprintf('The cost of the function is:%f\n ',cost);
fprintf('The Gradient of the function is:%f\n ',grad);
%Now optimizing the parameters usnig a built in function called fminunc
%setting up
options=optimset('GradObj','on','MaxIter',1000);
% [theta,cost]=...
%      fminsearch(@(t)(costFunction(t,X,y)),initial_theta,options);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
fprintf('The cost gained using fminunc is:%f\n',cost);
fprintf('The theta gained using fminunc:%f\n',theta);
%Now plot

%plotDecisionBoundary(theta, X, y);
%NOw we predict!!!
a1=input('How many times the female is pregnant? ');
a2=input('Plasma glucose concentration(a 2 hours in an oral test)e.g:148: ');
a3=input('Diastolic blood pressure(mm Hg)e.g 72: ');
a4=input('Triceps skinfold thickness(mm)e.g 35: ');
a5=input('2-Hour serum insulin(mu U/ml)e.g 0 or 94: ');
a6=input('Body mass index(weight(kg)/(height in m)^2)e.g 33.6: ');
a7=input('Diabetes pedigree function e.g 0.627: ');
a8=input('What is the Age? ');
prob=sigmoid([1 a1 a2 a3 a4 a5 a6 a7 a8 ]*theta);
fprintf('The probabilty of onset diabetes within next 5 years is:%f\n',prob);
%Now we compute the accuracy of our training set
p=predict(theta,X);
fprintf('Train Accuracy:%f\n',mean(double(p==y))*100);


