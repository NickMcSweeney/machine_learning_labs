function [J, grad] = costLogisticRegression(theta, X, y, lambda)
% Compute cost and gradient for logistic regression.

if nargin<4
    lambda = 0;
end

m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Compute the cost (J) and partial derivatives (grad) of a particular
% choice of theta. Make use the function sigmoid that you wrote earlier.
h = sigmoid(X*theta);
m = length(y);
K = length(theta);
lam = zeros(size(theta));
lam(1,[2:end]) = lambda;

J_t = (y'*log(h))+((1-y)'*log((1-h)));
grad_t = (h - y)'*X + lam'*theta;
%for i=1:m
    %J = J + (y(i)*log(h(i))+(1-y(i))*log(1-h(i)));
    %grad(1) = grad(1) + ((h(i)-y(i))*X(i,1));
    %for k=2:K
        %grad(k) = grad(k) + ((h(i)-y(i))*X(i,k)) + lambda*theta(k);
    %end
%end

%J = J/(-m) + (lambda/2)*sum(theta.*theta);
J = J_t/(-m) + (lambda/2)*(theta'*theta);

grad = grad_t./m;
% =============================================================

% unroll gradients
grad = grad(:);

end
