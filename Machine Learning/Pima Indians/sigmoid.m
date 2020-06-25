function g = sigmoid(z)

g = zeros(size(z));
k=1+exp(-z);
g=1./k;
end
