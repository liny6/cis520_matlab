function [error] = kernreg_xval_error(sigma, X, Y, part, distFunc)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(SIGMA, X, Y, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used (see KERNREG_TEST).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNREG_TEST

% FILL IN YOUR CODE HERE

N = max(part);

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
    S_i = length(actualTestLabels);
    predictedTestLabels = kernreg_test(sigma, trainPoints, trainLabels, testPoints, distFunc);
    %make binary classification
    predictedTestLabels = sign(predictedTestLabels);
    errors(i) = sum(predictedTestLabels ~= actualTestLabels)/S_i;
end

error = sum(errors)/N;