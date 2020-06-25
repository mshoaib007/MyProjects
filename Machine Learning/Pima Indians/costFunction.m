function [J,grad]=costFunction(theta,X,y)
m=length(y);
J=0;
grad=zeros(size(theta));
J=(-1/m)*sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta)));
tmp=sigmoid(X*theta);
error=(tmp-y);%error=(h of x -y)
grad=(1/m)*(X' *error);%(X' is x of j)

end

