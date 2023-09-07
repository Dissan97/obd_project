function mse = mse_loss(actual, predicted)
    % Ensure that actual and predicted have the same dimensions
    if numel(actual) ~= numel(predicted)
        error('Input vectors must have the same number of elements.');
    end
    
    % Calculate the squared differences between actual and predicted values
    squared_errors = (actual - predicted).^2;
    
    % Calculate the mean of squared errors to get the MSE
    mse = mean(squared_errors);
end