function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE
min_points = floor(n/n_folds);
mesh = meshgrid(1:n_folds, 1:min_points);
parts = mesh(:);
leftover = n - length(parts);
if leftover ~= 0
    parts = [parts; 1:leftover];
end

part = parts(randperm(n))';