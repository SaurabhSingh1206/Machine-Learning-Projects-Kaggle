function p=predict(all_theta,X)
  %%Predict the label for a trained 10 class classifier
  m=size(X,1);
  num_labels=size(all_theta,1);
  p=zeros(m,1);
  X=[ones(m,1) X];
[prob indices]=max((sigmoid(all_theta*X')),[],1);
p=indices';  
endfunction
