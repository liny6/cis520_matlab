%% Script/instructions on how to submit plots/answers for question 2.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data: this loads X, Xnoisy, and Y.
load('../data/breast-cancer-data-fixed.mat');

training_size = 400;

%{
%% 2.1
answers{1} = 'This is where your answer to 2.1 should go. Just as one long string in a cell array';

% Plotting with error bars: first, arrange your data in a matrix as
% follows:
%
%  nfold_errs(i,j) = nfold error with n=j of i'th repeat
nfold_errs = zeros(100, 4);
nfold_noisy_errs = zeros(100, 4);
K = 1;

for i = 1:100
    ind_training = (randperm(length(X)) <= training_size*ones(1,length(X)));
    X_training = X(ind_training, :);
    Y_training = Y(ind_training);
    X_testing = X(~ind_training,:);
    Y_testing = Y(~ind_training);
    
    X_noisy_training = X_noisy(ind_training, :);
    X_noisy_testing = X_noisy(~ind_training,:);
    
    
    
    for j = 1:4
        n_folds = 2^j;
        part = make_xval_partition(training_size, n_folds);
        
        nfold_errs(i, j) = knn_xval_error(K, X_training, Y_training, part);
        nfold_noisy_errs(i, j) = knn_xval_error(K, X_noisy_training, Y_training, part);
        %this part finds test error
        
    end
end

figure(1)
bar(nfold_errs);
xlabel('iteration')
ylabel('error')
legend('2 fold', '4 fold', '8 fold', '16 fold')
title('in-fold error, original data')

figure(2)
bar(nfold_noisy_errs);
xlabel('iteration')
ylabel('error')
legend('2 fold', '4 fold', '8 fold', '16 fold')
title('in-fold error, noisy data')
%  
% Then we want to plot the mean with error bars of standard deviation as
% folows: y = mean(nfold_errs), e = std(nfold_errs), x = [2 4 8 16].
% 
% >> errorbar(x, y, e);
%
y = mean(nfold_errs);
e = std(nfold_errs);
x = [2 4 8 16];

figure(3)
errorbar(x, y, e);
xlabel('n folds')
ylabel('error')
title('average in-fold error')

y_noisy = mean(nfold_noisy_errs);
e_noisy = std(nfold_noisy_errs);

figure(4)
errorbar(x, y_noisy, e_noisy);

% Along with nfold_errs, also plot errorbar for test error. This will 
% serve as measure of performance for different nfold-crossvalidation.
%

K = 1:2:15;
sigma = 1:8;
knn_tenfold_errs = zeros(100,8);
knn_tenfold_noisy_errs = zeros(100,8);
ker_tenfold_errs = zeros(100,8);
ker_tenfold_noisy_errs = zeros(100,8);
knn_test_errs = zeros(100,8);
knn_test_noisy_errs = zeros(100,8);
ker_test_errs = zeros(100,8);
ker_test_noisy_errs = zeros(100,8);

%}

for i = 1:100
    ind_training = (randperm(length(X)) <= training_size*ones(1,length(X)));
    X_training = X(ind_training, :);
    Y_training = Y(ind_training);
    X_testing = X(~ind_training,:);
    Y_testing = Y(~ind_training);
    
    X_noisy_training = X_noisy(ind_training, :);
    X_noisy_testing = X_noisy(~ind_training,:);
    
    for j = 1:8
        part = make_xval_partition(training_size, 10);%new partrition
        
        knn_tenfold_errs(i, j) = knn_xval_error(K(j), X_training, Y_training, part); %
        knn_tenfold_noisy_errs(i,j) = knn_xval_error(K(j), X_noisy_training, Y_training, part); 
        
        knn_test_no_match = (sign(knn_test(K(j), X_training, Y_training, X_testing)) ~= sign(Y_testing));
        knn_test_errs(i, j) = sum(knn_test_no_match);
        knn_test_noisy_no_match = (sign(knn_test(K(j), X_noisy_training, Y_training, X_testing)) ~= sign(Y_testing));
        knn_test_noisy_errs(i, j) = sum(knn_test_noisy_no_match);
        
        ker_tenfold_errs(i, j) = kernreg_xval_error(K(j), X_training, Y_training, part);
        ker_tenfold_noisy_errs(i, j) = kernreg_xval_error(K(j), X_noisy_training, Y_training, part);
        
        ker_test_no_match = (sign(kernreg_test(K(j), X_training, Y_training, X_testing)) ~= sign(Y_testing));
        ker_test_errs(i, j) = sum(ker_test_no_match);
        ker_test_noisy_no_match = (sign(kernreg_test(K(j), X_noisy_training, Y_training, X_testing)) ~= sign(Y_testing));
        ker_test_noisy_errs(i, j) = sum(ker_test_no_match);
    end
    
end

y_knn_tenfold = mean(knn_tenfold_errs);
e_knn_tenfold = std(knn_tenfold_errs);
y_knn_test = mean(knn_test_errs);
e_knn_test = std(knn_tenfold_errs);
y_ker_tenfold = mean(ker_tenfold_errs);
e_ker_tenfold = std(ker_tenfold_errs);
y_ker_test = mean(ker_test_errs);
e_ker_test = std(ker_test_errs);

y_knn_tenfold_noisy = mean(knn_tenfold_noisy_errs);
e_knn_tenfold_noisy = std(knn_tenfold_noisy_errs);
y_knn_test_noisy = mean(knn_test_noisy_errs);
e_knn_test_noisy = std(knn_test_noisy_errs);
y_ker_tenfold_noisy = mean(ker_tenfold_noisy_errs);
e_ker_tenfold_noisy = std(ker_tenfold_noisy_errs);
y_ker_test_noisy = mean(ker_test_noisy_errs);
e_ker_test_noisy = std(ker_test_noisy_errs);

figure(5) %knn X-validation
errorbar(K, y_knn_tenfold, e_knn_tenfold);

figure(6) %knn test
errorbar(K, y_knn_test, e_knn_test);

figure(7) %ker X-validation
errorbar(sigma, y_ker_tenfold, e_ker_tenfold);

figure(8) %ker test
errorbar(sigma, y_ker_test, e_ker_test);

figure(9) %knn X-validation noisy
errorbar(K, y_knn_tenfold_noisy, e_knn_tenfold_noisy);

figure(10) %knn test noisy
errorbar(K, y_knn_test_noisy, e_knn_test_noisy);

figure(11) %ker X-validation noisy
errorbar(sigma, y_ker_tenfold_noisy, e_ker_tenfold_noisy);

figure(12) %ker test
errorbar(sigma, y_ker_test_noisy, e_ker_test_noisy);


% To add labels to the graph, use xlabel('X axis label') and ylabel
% commands. To add a title, using the title('My title') command.
% See the class Matlab tutorial wiki for more plotting help.
% 
% Once your plot is ready, save your plot to a jpg by selecting the figure
% window and running the command:
%
% >> print -djpg plot_2.1-noisy.jpg % (for noisy version of data)
% >> print -djpg plot_2.1.jpg  % (for regular version of data)
%
% YOU MUST SAVE YOUR PLOTS TO THESE EXACT FILES.

%% 2.2
answers{2} = 'This is where your answer to 2.2 should go. Short and sweet is the key.';

% Save your plots as follows:
%
%  noisy data, k-nn error vs. K --> plot_2.2-k-noisy.jpg
%  noisy data, kernreg error vs. sigma --> plot_2.2-sigma-noisy.jpg
%  regular data, k-nn error vs. K --> plot_2.2-k.jpg
%  regular data, kernreg error vs. sigma --> plot_2.2-sigma.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_2_answers.mat', 'answers');
