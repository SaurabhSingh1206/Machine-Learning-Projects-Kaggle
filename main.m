%%This Problem has been taken from kaggle(link:https://www.kaggle.com/zalando-research/fashionmnist/home)
%%It is a simple classification problem wherein we are required to classify articles based on their grayscale 
%%images into 10 different categories.

clear;close all;clc
input_layer_size=784;
num_labels=10;
%%======================Loading Data==============================%%
fprintf("Loading Data.\nThis may take a while. Please wait...");
train_data=load('trainset.m');
test_data=load('testset.m');
y=train_data(:,1);
ytest=test_data(:,1);
X=train_data(:,2:end);
Xtest=test_data(:,2:end);

%%Replacing class label 0 with 10
for i=1:size(y,1),
  if y(i,1)==0,
    y(i,1)=10;
   endif;
endfor;
for i=1:size(ytest,1),
  if ytest(i,1)==0,
    ytest(i,1)=10;
   endif;
endfor;
%%=======================Training Parameters============================%%
fprintf('\nTraining parameters. Please wait...\n');
%This particular value of lambda=30 was selected after examining learning 
%curves(i.e. plots of J(train) & J(cv) v/s m;and validation curve(i.e. J(train) & J(cv) v/s lambda)
lambda=30;
[theta]=Classifier(X,y,num_labels,lambda);
fprintf('Parameters obtained. Press enter to know the prediction accuracy\n');
pause;



%%=======================Prediction stage==============================%%
pred=predict(theta,X);
fprintf('\n Training Set accuracy: %f\n',mean(double(pred==y))*100);
pred=predict(theta,Xtest);
fprintf('\n Test Set accuracy: %f\n',mean(double(pred==ytest))*100);

