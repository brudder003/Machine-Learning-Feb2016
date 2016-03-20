function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


theta_s = theta(2:size(theta));
theta_reg = [0;theta_s];

% cost and grandient just like before
cost = -1/m*(log((sigmoid(X*theta))')*y+log((1-sigmoid(X*theta))')*(1-y));
grad = (1/m).*X'*(sigmoid(X*theta)-y);

% compute the regularized cost
J = cost + (lambda / (2*m)) * sum(theta_reg .^ 2);

% compute regularized grad
reg_vector = (lambda / m) * theta;
% set first row to zero and add this to grad
reg_vector(1) = 0;

%add the gradient and the regularization vector
grad = grad + reg_vector;

% =============================================================

end
