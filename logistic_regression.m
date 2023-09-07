function [w, b] = logistic_regression(x_train, y_train,num_iterations, lambda, alpha)

    % Initialize parameters
    n = size(x_train, 2); % Number of features
    w = rand(n, 1) - 0.5; % Initialize weights
    b = rand(1) - 0.5; % Initialize bias

    for iter = 1:num_iterations
        % Compute the logistic regression hypothesis
        z = x_train * w + b;
        h = sigmoid(z);
        
        % Compute the gradient of the log-likelihood with regularization
        gradient_w = (x_train' * (h - y_train) + 2 * lambda * w) / length(y_train);
        gradient_b = sum(h - y_train) / length(y_train);
        
        % Update parameters
        w = w - alpha * gradient_w;
        b = b - alpha* gradient_b;
    end

end

