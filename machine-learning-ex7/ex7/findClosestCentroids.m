function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%there are 300 examples 
m = size(X,1);

#go over each example one by one
for i = 1:m
  
  #pull out numbers for the example being looped over
  xi = X(i,:);
  
  #initialize zeros the same number as centroids
  minu = zeros(size(centroids,1),1);
  
  #go over each cluster centroid
  for j = 1:K
    
    #pull out centroid location
    ui = centroids(j,:);
    
    #calc distance btwn example and each centroid
    dist = norm(xi-ui)^2;
    
    #put the value in minu vector
    minu(j) = dist;
    
   end
    
   #there should be three ci's in the minu vector
   #take the smallest and put it in idx
   [minval, ci] = min(minu);
   
   idx(i) = ci;
   
 end

% =============================================================

end

