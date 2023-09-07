function [y_pred] = forward_data(w, b,activation_functions, x_t, y_t)
    n = size(x_t, 1);
    y_pred = zeros(n, 1);
    losses = zeros(n, 1);
    
        for sample=1:n
            x = x_t(sample, :);
            activations = cell(1, length(w) + 1);
            z_values = cell(1, length(w));
            activations{1} = x';
            % forward pass
            for i = 1:length(w)
                z_values{i} = w{i} * activations{i} + b{i};
    
                % activation function
                if strcmp(activation_functions{i}, 'relu')
                    activations{i + 1} = relu(z_values{i});
                elseif strcmp(activation_functions{i}, 'sigmoid')
                    activations{i + 1} = sigmoid(z_values{i});
                end
            end
    
            % calculate loss function 
            loss = binary_cross_entropy(y_t(sample), activations{end});
            losses(sample) = loss;
            y_pred(sample) = activations{end};
        end
        % Display training and validation metrics

    [accuracy, recall, precision] = calculate_metrics(y_t, y_pred);
    fprintf("precision: %f - recall: %f - accuracy: %f\n" + ...
        "loss: %f - EmpiricalRisk: %f\n", accuracy, recall, precision, mean(losses), sum(losses)/n);
end

