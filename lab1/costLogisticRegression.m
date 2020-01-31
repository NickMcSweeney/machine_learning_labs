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
J = ...
grad = ...
% =============================================================

% unroll gradients
grad = grad(:);

end
