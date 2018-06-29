function [J, grad]=CostFunction(theta,X,y,lambda)
  %%This function computes the cost and the gradient for logistic regression
  m=length(y);
  J=0;
  grad=zeros(size(theta));
  temp=theta;
  temp(1)=0;
  J=-(1/m)*sum(y.*log(sigmoid(X*theta))+(ones(m,1)-y).*log(ones(m,1)-sigmoid(X*theta)))+(lambda/(2*m))*sum(temp.^2);
  grad=(1/m)*((sigmoid(X*theta)-y)'*X)'+(lambda/m)*temp;
endfunction
