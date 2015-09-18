%% Script/instructions on how to submit plots/answers for question 3.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data
data = load('../data/mnist_all.mat');

% Running a training set for binary decision tree classifier
%[X Y] = get_digit_dataset(data, {'7','9'}, 'train');

%% Train a depth 4 binary decision tree
%dt = dt_train(X, Y, 4);

%%
%[Xtest Ytest] = get_digit_dataset(data, {'7','9'}, 'test');

%Yhat = zeros(size(Ytest));
%for i = 1:size(Xtest,1)
%    Yhat(i) = dt_value(dt, Xtest(i,:)) >= 0.5;
%end

%mean(Yhat ~= Ytest)

%% 3.1
answers{1} = 'This is where your answer to 3.1 should go. Just as one long string in a cell array';

%pull training set
[Xtrain1, Ytrain1] = get_digit_dataset(data, {'1','3','7'},'train');
%pull testing set
[Xtest1, Ytest1] = get_digit_dataset(data, {'1','3','7'},'test');
Y_est1_test = zeros(size(Ytest1));
Y_est1_training = zeros(size(Ytrain1));
training_error= zeros(6,1);
testing_error = zeros(6,1);
%Train decision trees of various depth
for i = 1:6
    dt = dt_train_multi(Xtrain1, Ytrain1, i);
    
    for j = 1:size(Xtest1, 1)
        [~, Y_est1_test(j)] = max(dt_value(dt, Xtest1(j,:)));
    end
    
    for j = 1:size(Xtrain1, 1)
        [~, Y_est1_training(j)] = max(dt_value(dt, Xtrain1(j,:)));
    end    
    
    testing_error(i) = mean(Y_est1_test ~= Ytest1);
    training_error(i) = mean(Y_est1_training ~= Ytrain1);
end

%plot depth vs training/ testing error
plot(1:6, training_error)
hold on
plot(1:6, testing_error)





% Saving your plot: once you have succesfully plotted your data; e.g.,
% something like:
% >> plot(depth, [train_err test_err]);
% Remember: You can save your figure to a .jpg file as follows:
% >> print -djpg plot_3.1.jpg

%% 3.2
answers{2} = 'This is where your answer to 3.2 should go. Short and sweet is the key.';

% Saving your plot: once you've computed M, plot M with the plotnumeric.m
% command we've provided. e.g:
% >> plotnumeric(M);
%
% Save your file to plot_3.2.jpg
%
% ***** ALSO *******
% Save your confusion matrix M to a .txt file as follows:
% >> save -asci confusion.txt M

%% 3.3
answers{3} = 'This is where your answer to 3.3 should go. Please be concise.';

% E.g., if Xtest(i,:) is an example your method fails on, call:
% >> plot_dt_digit(tree, Xtest(i,:));
%
% Save your file to plot_3.3.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_3_answers.mat', 'answers');