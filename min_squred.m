function [w, b] = min_squred(x_train, y_train, lambda, max_iters, alpha)
    
m = size(x_train, 1); % Number of training examples
n = size(x_train, 2); % Number of features
w = zeros(n, 1); % Initialize w
b = 0; % Initialize b
tolerance = 1e-6; % Convergence tolerance

for iter = 1:max_iters
    % Update w using Gauss-Seidel coordinate descent
    for j = 1:n
        gradient_w_j = -2 * sum((y_train - (x_train * w + b)) .* x_train(:, j)) / m + 2 * lambda * w(j);
        w(j) = w(j) - alpha * gradient_w_j;
    end
    
    % Update b using Gauss-Seidel coordinate descent
    gradient_b = -2 * sum(y_train - (x_train * w + b)) / m;
    b = b - alpha * gradient_b;
    
    % Check for convergence
    if norm([gradient_w_j; gradient_b]) < tolerance
        fprintf('Converged after %d iterations\n', iter);
        break;
    end

    if mod(iter, 100) == 0
        fprintf("mse iteration: %d\n", iter);
    end
end
    

end

