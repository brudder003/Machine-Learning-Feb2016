
clear ; close all; clc
grad = zeros(size(theta));


X = load('logistic_x.txt');
y = load('logistic_y.txt');

for i=1:99
  if (y(i) == -1);
    y(i) = 0;
  endif
end

for i=1:3
    grad(i) = grad(i) + (-1/m)*(1-sigmoid(y(i)*X(i,:)*theta))*y(i)*X(i,:);
end



[m, n] = size(X);
X = [ones(m,1) X];
initial_theta = zeros(n+1,1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;