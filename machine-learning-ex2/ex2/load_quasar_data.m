% load_quasar_data
%
% Loads the data in the quasar data files
%
% Upon completion of this script, the matrices and data are as follows:
%
% lambdas - A length n = 450 vector of wavelengths {1150, ..., 1599}
% train_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy
%      observed quasar spectra for training.
% test_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy observed
%       quasar spectra for testing.

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);

%[m, n] = size(train_qso);
%train_qso_first = [ones(1,length(train_qso)) ; train_qso(1,:)]';


X = [ones(length(lambdas),1) lambdas];
y_full = train_qso;
[m,n] = size(y_full);

y_smooth = zeros(n, m);

for i = 1:n
    y = y_full(i,:)';
    for j = 1:m
        w_j = e.^(-(X(j,2) - X(:,2)).^2/(2*5^2));
        W = diag(w_j);
        theta = pinv(X'*W*X)*X'*W*y(j,:);
        y_j(i,j) = X(j,:)*theta;
    end
   y_smooth(i,:) = y_j;
end