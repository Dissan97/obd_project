function [loss_prime] = binary_cross_entropy_prime(y_true, y_pred)
    loss_prime = (y_pred - y_true) ./ (y_pred .* (1 - y_pred));
end

