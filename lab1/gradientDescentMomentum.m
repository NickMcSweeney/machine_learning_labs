function [theta J_history] = gradientDescentMomentum(theta, X, y, alpha, num_iters)
% Runs gradient descent.

J_history = zeros(num_iters, 1);
v = 0;
for iter = 1:num_iters
    [J grad] = costLinearRegression(theta, X, y);
    J_history(iter) = J;

    % ====================== YOUR CODE HERE ======================
    % Update the parameter vector theta by using alpha and grad.
    gamma = 0.9;
    if iter < 5
        gamma = 0.5;
    end
    v = gamma*v + alpha*grad;
    theta = theta - v;
    % ============================================================

end

end
