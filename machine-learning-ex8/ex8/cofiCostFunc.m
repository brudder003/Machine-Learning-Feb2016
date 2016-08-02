function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
%unvectorized, i did this first bc it seemed easier,
%really vectorized was even easier to think about,
%maybe the matrix manipulation is starting to stick :)

%costij =0;
%cost = 0;

%for i=1:size(R,1)
 % for j=1:size(R,2)
  %  if R(i,j) == 1
   %   costij = (Theta(j,:)*X(i,:)' - Y(i,j))^2;
    %  cost = cost + costij;
    %end
  %end
%end

%COST FUNCTION
M = zeros(size(R));
%VECTORIZED
M = (X*Theta' - Y).^2;
%compute cost, I calculed the cost for all obs
%bc vectorized is faster than loops,
%but now need to elimated all the costs i calc-ed when
% a movie was rated, ie r(i,j) = 0

%regularization term
reg = 0.5*lambda*sum(sum(Theta.^2)) + 0.5 * lambda*(sum(sum(X.^2)));

J = (0.5)*(sum(sum(R.*M))) + reg;

%Gradient
X_grad = R.*(X*Theta' - Y) * Theta + lambda*X;

Theta_grad = (R.*(X*Theta' - Y))' * X + lambda*Theta;
  

%XM = zeros(size(Theta));
%XM = (X*Theta' - Y);



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
