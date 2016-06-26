function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X is 5000 x 400 right now, need to add ones to the beginning
% for the bias node
X = [ones(m,1) X];


%feedforward, cost back prop
for i = 1:m
  
  %FEEDFORWARD

  % X is 5000x401 going through 1 pic at a time
  a1 = X(i,:);
  % theta1 is 25x401 a1 is 1x401 
  z2 = Theta1*a1';
  % a2 is sigmoid of z2 and add the ones for the next layer
  a2 = sigmoid(z2);
  a2 = [1; a2];
  % calc z3 for the output layer
  z3 = Theta2*a2;
  % a3 (output) is sigmoid z3
  a3 = sigmoid(z3);

  % for the given i change y into a vector that is zero
  % everywhere except its the index of its correct labels
  y_label = zeros(num_labels,1);
  y_label(y(i)) = 1;
  
  %BACKPROPAGATION
   
  %for each output unit k in layer 3 set d3=(a3 - y)
  d3 = a3 - y_label;
  
  %For the hidden layer l=2 set d2 = t2'*d3.*sigmoidGrad(z2)
  z2 = [1;z2];
  d2 = Theta2'*d3.*sigmoidGradient(z2);
  d2 = d2(2:end);
  
  %accumulate the gradient from this example
  Theta1_grad = Theta1_grad + d2*a1;
  Theta2_grad = Theta2_grad + d3*a2';
  
  cost = 0;
  % another for loop to add up cost by labels
  for k = 1:num_labels
    costk = -1/m*(y_label(k)*log(a3(k)) + (1-y_label(k))*log(1-a3(k)));
    cost = cost + costk;
  end
  
  %add the cost form that observation to the rest of the cost
  J = J + cost;
  
 end
 
 %dont want to use the bias unit in regularization
 %regularize by taking the square of each element in theta1
 %then summing theta element by element
 Theta1_reg = Theta1(:,2:end);
 Theta1_reg = Theta1_reg.^2;
 T1_regsum= sum(Theta1_reg(:));
 
 Theta2_reg = Theta2(:,2:end);
 Theta2_reg = Theta2_reg.^2;
 T2_regsum = sum(Theta2_reg(:));
 
 %add the two thetas assuming we have one hidden layer
 %but the program works for any size theta
 Reg = (lambda / (2*m)) * (T2_regsum+T1_regsum);
 
 %add the reg term to the rest of our cost
 J = J + Reg;
 
 %gradients
 Theta1_grad = (1/m)*Theta1_grad +(lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
 Theta2_grad = (1/m)*Theta2_grad +(lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
 
   
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
