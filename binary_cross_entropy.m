function [loss] = binary_cross_entropy(y_true,y_pred)
    
    epsilon = 1e-15;
    loss = -sum(y_true .* log(y_pred + epsilon) + (1 - y_true) .* log(1 - y_pred + epsilon));
end

