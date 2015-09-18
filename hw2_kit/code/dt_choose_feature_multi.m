function [fidx, val, max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

%take Z value to true and falses to probability distribution
Z_distr = sum(Z)./sum(sum(Z))';
%get empirical entropy for the Z distribution
H_z = multi_entropy(Z_distr);

ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', length(Z));

for i = colidx
    t.timeleft();
    
    %check for constant values in X
    if numel(Xrange{i}) == 1
        ig(i) = 0; split_vals(i) = 0;
        continue;
    end
    
    %otherwise, split the feature up to 10 bins and try to find the best
    %split value
    r = linspace(double(Xrange{i}(1)), double(Xrange{i}(end)), min(10, numel(Xrange{i})));
    split_compare = bsxfun(@le, X(:,i), r(1:end-1));
    px = mean(split_compare);
    cond_H_z = zeros(1,size(split_compare,2));
    
    
    for j = 1:size(split_compare,2)
        z_given_x = bsxfun(@and, Z, split_compare(:,j));
        z_given_not_x = bsxfun(@and, Z, ~split_compare(:,j));
        z_distr_given_x = sum(z_given_x)./sum(sum(z_given_x))';
        z_distr_given_not_x = sum(z_given_not_x)./sum(sum(z_given_not_x))';
        cond_H_z(j) = px(j).*multi_entropy(z_distr_given_x') + (1-px(j)).*multi_entropy(z_distr_given_not_x');
    end
    
    [ig(i), best_split] = max(H_z-cond_H_z);
    split_vals(i) = r(best_split); 
end

[max_ig, fidx] = max(ig);
val = split_vals(fidx);


% YOUR CODE GOES HERE