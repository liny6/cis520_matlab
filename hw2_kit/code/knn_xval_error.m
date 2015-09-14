function [error] = knn_xval_error(K, X, Y, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(K, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KNN_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KNN_TEST

% FILL IN YOUR CODE HERE


if nargin<5
    distFunc = 'l2';
end

folds = max(part);
errors = zeros(folds, 1);

for i = 1:folds
    trainPoints = X(part ~= i, :);
    trainLabels = Y(part ~= i);
    testPoints = X(part == i, :);
    actualTestLabels = Y(part == i);
    predictedTestLabels = knn_test(K, trainPoints, trainLabels, testPoints, distFunc);
    %make binary classification
    predictedTestLabels = sign(predictedTestLabels);
    errors(i) = sum(predictedTestLabels ~= actualTestLabels);
end

error = mean(errors);