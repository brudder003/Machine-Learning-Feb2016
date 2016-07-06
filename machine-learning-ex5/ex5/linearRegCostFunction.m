function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
cost=0;
reg=0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%make theata zero equal to zero so it doesnt contriubute to the sum;
theta_s = theta(2:size(theta));
theta_reg = [0;theta_s];


%vectorized implempentation of linear regression cost function
cost = 1/(2*m)*((X*theta - y)'*(X*theta - y));

reg = (lambda / (2*m)) * sum(theta_reg .^ 2);

J=cost + reg;

%vectorized implementation of the gradient
grad = (1/m).*X'*((X*theta)-y);

% compute regularized grad
reg_vector = (lambda / m) * theta;
% set first row to zero and add this to grad
reg_vector(1) = 0;

%add the gradient and the regularization vector
grad = grad + reg_vector;



% =========================================================================

grad = grad(:);

end
