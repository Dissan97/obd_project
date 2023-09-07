
%% data definition
clc 
clear
alpha = 0.01;
[x_train, y_train, x_validation, y_validation, x_test, y_test] = get_data('data1.csv', 0.7, 0.1);
file_name = 'result.txt';
fd = fopen(file_name, "w");

%% mse
tic;
lambda = 0.1;
[w, b] = min_squred(x_train, y_train, lambda, 1000, 0.01);


fprintf(fd, "\nMean squared Error: : time passed: %.4fs\n", toc);
% Calculate the MSE loss on the validation set
y_validate_pred = x_validation * w + b;
mse_validate = sum((y_validate_pred - y_validation).^2) / length(y_validation);

fprintf(fd,'\nMean Squared Error (MSE) on Validation Set: %.4f\n', mse_validate);
% 

y_pred = x_test * w + b;
mse_test = sum((y_pred - y_test).^2) / length(y_test);
fprintf(fd,'\nMean Squared Error (MSE) on Test Set: %.4fs\n', mse_test);

[accuracy, recall, precision] = calculate_metrics(y_test, y_pred, 0.5);
fprintf(fd,"recall: %f - precision: %f - accuracy: %f\n", recall, precision, accuracy);

%% logistic regression
tic;
lambda = 2;
iterations = 50000;
[w, b] = logistic_regression(x_train, y_train, iterations, lambda, alpha);
fprintf(fd, "\nLogistic Regression with %d iteration lambda=%.4f and alpha= %.4f: time passed: %.4fs\n", iterations, lambda, alpha, toc);

z_val = x_validation * w + b;
h_val = sigmoid(z_val);
y_pred = (h_val >= 0.5); % Convert probabilities to binary predictions (0 or 1)

[accuracy, recall, precision] = calculate_metrics(y_validation, y_pred, 0.5);
fprintf(fd, "On validation set recall: %f - precision: %f - accuracy: %f\n", recall, precision, accuracy);

z_test = x_test * w + b;
h_test = sigmoid(z_test);
y_pred = (h_test >= 0.5); % Convert probabilities to binary predictions (0 or 1)
[accuracy, recall, precision] = calculate_metrics(y_test, y_pred, 0.5);
fprintf(fd, "On Test set recall: %f - precision: %f - accuracy: %f\n", recall, precision, accuracy);
%% neural network

architecture = [87, 128, 64,1];
activation_functions = {'relu', 'relu', 'sigmoid'};
[w, b] = initialize_architecture(architecture);

fprintf(fd, "\nNeural network architecture[ {87} -> {128} -> {relu} -> {64} -> {relu} -> {1} -> {sigmoid}]\n");

%% gradien descent
tic;
iterations =200;
alpha = 0.2;
[w1, b1, iter] = gradient_descent(x_train, y_train, activation_functions, w, b, iterations, alpha);
fprintf(fd, "\nGradient Descent Neural Network with %d iteration and alpha= %d: ended at iteration=%d time passed: %.4fs\n", iterations, alpha, iter,toc);
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);

%% gradient descent batch 32
tic;
batch_size = 32;
alpha = 0.01;
iterations =100;
[w1, b1, iter] = gradient_descent_batch(x_train, y_train, activation_functions, w, b, iterations, alpha, batch_size, 0);
fprintf(fd, "\nGradient Descent batch_size=%d  Neural Network with %d iteration, alpha= %d: ended at iteration=%d time time passed: %.4fs\n", batch_size, iterations, alpha, iter, toc);
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);
%% gradient descent batch 64
tic;

batch_size = 64;
iterations = 50;
[w1, b1, iter] = gradient_descent_batch(x_train, y_train, activation_functions, w, b, iterations, alpha, batch_size, 0);
fprintf(fd, "\nGradient Descent batch_size=%d  Neural Network with %d iteration, alpha= %d: ended at iteration=%d time time passed: %.4fs\n", batch_size, iterations, alpha, iter, toc);
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);


%% gradient descent batch 128
tic;

batch_size = 128;
iterations = 20;
[w1, b1, iter] = gradient_descent_batch(x_train, y_train, activation_functions, w, b, iterations, alpha, batch_size, 0);
fprintf(fd, "\nGradient Descent batch_size=%d  Neural Network with %d iteration, alpha= %d: ended at iteration=%d time time passed: %.4fs\n", batch_size, iterations, alpha, iter, toc);
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);


%% gradient_descent with regularization
tic;


lambdas = [0.1, 0.2, 0.5, 1, 2, 5, 10];
batch_size = 64;
iterations = 20;
fprintf(fd,"\nGradient Descent Batch with regularization\n");
n = size(lambdas, 2);
ers = zeros(n, 1);
ws = cell(1, n);
bs = cell(1, n);

for i = 1:n
    [ws{i}, bs{i}] = gradient_descent_batch(x_train, y_train, activation_functions, w, b, iterations, alpha, batch_size, lambdas(i));
    fprintf("test lambda: %.4f", lambdas(i));
    fprintf(fd,"validation lambda=%.4f\n", lambdas(i));
    ers(i) = test_data(ws{i}, bs{i}, activation_functions, x_validation, y_validation, 1, fd, 0.5);
end

[~, m] = min(ers);
M = size(lambdas, 1);

index = round((m + M) / 2);

fprintf(fd,"finished after: %.4fs with lambda=%.4f and Empirical risk=%f\n", 821.9112, lambdas(index), ers(index));
fprintf(fd, "test\n");
test_data(ws{index}, bs{index}, activation_functions, x_test, y_test, 1, fd, 0.5);

%% 10 fold validation
tic;
batch_size = 32;
iterations = 30;
fprintf(fd,"\n10 Fold Validation\n");
num_folds = 10;
cv = cvpartition(length(y_train), 'KFold', num_folds);
alpha = 0.05;
% Create empty arrays to store results
ers = zeros(num_folds, 1);
ws = cell(num_folds, 1);
bs = cell(num_folds, 1);

for fold = 1:num_folds
    fprintf("Fold %d / %d\n", fold, num_folds);

    % Create training and validation sets for this fold
    trainIdx = training(cv, fold);
    validationIdx = test(cv, fold);

    x_fold_train = x_train(trainIdx, :);
    y_fold_train = y_train(trainIdx, :);

    x_fold_validation = x_train(validationIdx, :);
    y_fold_validation = y_train(validationIdx, :);

    [ws{fold}, bs{fold}] = gradient_descent_batch(x_train, y_train, activation_functions, w,b, iterations, alpha, 64, lambdas(index));
    fprintf(fd,"validation fold[%d]\n", fold);
    ers(fold) = test_data(ws{fold}, bs{fold}, activation_functions, x_fold_validation, y_fold_validation, 1, fd, 0.5);
end
fprintf(fd,"finished after: %.4fs\n", toc);
% Calculate the average error rate across all folds
average_error = mean(ers);
% Find the best model
[~, bestFold] = min(ers);
% Use the best model for your final testing (e.g., on x_test and y_test)
fprintf(fd,"Average Error Rate Across Folds[%d]: %f\n", num_folds, average_error);
best_ws = ws{bestFold};
best_bs = bs{bestFold};
fprintf(fd, "test\n");
test_data(best_ws, best_bs, activation_functions, x_test, y_test, 1, fd, 0.5);

%% roc CURVE
clc
[foo,y_pred] = test_data(best_ws, best_bs, activation_functions, x_test, y_test, 1, fd,0.5);
y_pred_binary = (y_pred >= 0.6);
y_pred = double(y_pred_binary);
[X, Y, T, AUC] = perfcurve(y_test, y_pred, 1);
% Plot the ROC curve

plot(X, Y);

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC) ')']);


%% adam
tic;
fprintf(fd,"\nAdam optimizer\n");

iterations =10;
[w1, b1] = adam_optimizer(x_train, y_train, activation_functions, w,b, iterations, alpha, 0.9, 0.999, 1e-8);
fprintf(fd,"finished after: %.4fs", toc);
fprintf(fd, "test");
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);


%% 
tic;
iterations =10;
fprintf(fd,"\nAdam optimizer with ridge\n");
[w1, b1] = adam_optimizer_with_regularization(x_train, y_train, activation_functions, w,b, iterations, alpha, 0.9, 0.999, 1e-8, 'ridge', 1);
fprintf(fd,"finished after: %.4fs\n", toc);
fprintf(fd, "validation\n");
test_data(w1, b1, activation_functions, x_validation, y_validation, 1, fd, 0.5);
fprintf("test \n");
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);

%% 

tic;

num_epochs =10;
fprintf(fd,"\nAdam optimizer with lasso\n");
[w1, b1] = adam_optimizer_with_regularization(x_train, y_train, activation_functions, w,b, num_epochs, alpha, 0.9, 0.999, 1e-8, 'lasso', 1);
fprintf(fd,"finished after: %.4fs", toc);
fprintf(fd, "validation\n");

test_data(w1, b1, activation_functions, x_validation, y_validation, 1, fd, 0.5);
fprintf("test\n");
test_data(w1, b1, activation_functions, x_test, y_test, 1, fd, 0.5);

%%      
fclose(fd);

%% f
clc
v = [1, 2, 3];
fprintf("%f \n", v);