function [all_theta]=Classifier(X,y,num_labels,lambda)
  [m,n]=size(X);
  all_theta=zeros(num_labels,n+1);
  X=[ones(m,1) X];
  initial_theta=zeros(n+1,1);
  options=optimset('GradObj','on','MaxIter',50);
  for c=1:num_labels,
    [theta]=fmincg(@(t)(CostFunction(t,X,(y==c),lambda)),initial_theta,options);
    all_theta(c,:)=theta';
  endfor;  
endfunction
