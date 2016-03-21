function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X is 5000 x 400 right now, need to add ones to the beginning
% for the bias node
a1 = [ones(m,1) X];

% just following the diagram in the homework, calculate z for the 
% hidden layer a1 is 5000x401 theta1 is 25x401 need to transpose
% to multiply a1*theta'
z2 = a1*Theta1';

% a2 is sigmoid of z2 and add the ones for the next layer
a2 = [ones(size(z2),1) sigmoid(z2)];

% calc z3 for the output layer
z3 = a2*Theta2';

% a3 (output) is sigmoid z3
a3 = sigmoid(z3);

%obtain max for each row like the one v all method
[predict_max, index_max] = max(a3, [], 2);

%return p
p = index_max;



% =========================================================================


end
