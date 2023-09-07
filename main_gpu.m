%% data definition
clc 
clear
alfa = 0.01;
[x_train, y_train, x_validation, y_validation, x_test, y_test] = get_data('winequality-white.csv', 0.7, 0.1, 'quality');

%%
clc
tic;
iterations =200;
alpha = 0.2;
architecture = [11, 128, 64,1];
activation_functions = {'relu', 'relu', 'sigmoid'};
[w, b] = initialize_architecture(architecture);
[w1, b1, iter] = gradient_descent(x_train, y_train, activation_functions, w, b, iterations, alpha, 'mse');
fprintf("\nGradient Descent Neural Network with %d iteration and alpha= %d: ended at iteration=%d time passed: %.4fs\n", iterations, alpha, iter,toc);
