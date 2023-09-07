function [weights, biases, epoch] = gradient_descent(x_train, y_train, activation_functions, weights, biases, num_epochs, alfa)

    n = size(x_train, 1);
    y_pred = zeros(n, 1);
    losses = zeros(n, 1);
    for epoch=1:num_epochs
    
        for sample=1:n
            x = x_train(sample, :);
            activations = cell(1, length(weights) + 1);
            z_values = cell(1, length(weights));
            activations{1} = x';
            % forward pass
            for i = 1:length(weights)
                z_values{i} = weights{i} * activations{i} + biases{i};
    
                % activation function
                if strcmp(activation_functions{i}, 'relu')
                    activations{i + 1} = relu(z_values{i});
                elseif strcmp(activation_functions{i}, 'sigmoid')
                    activations{i + 1} = sigmoid(z_values{i});
                end
            end
    
            % calculate loss function 

            loss = binary_cross_entropy(y_train(sample), activations{end});

            losses(sample) = loss;
            y_pred(sample) = activations{end};
            % backward pass
    
            deltas = cell(1, length(weights));
            deltas{end} = activations{end} - y_train(sample);
            for i = length(weights) - 1:-1:1
                if strcmp(activation_functions{i}, 'relu')
                    deltas{i} = (weights{i + 1}' * deltas{i + 1}) .* double(z_values{i} > 0); % ReLU gradient
                elseif strcmp(activation_functions{i}, 'sigmoid')
                    deltas{i} = (weights{i + 1}' * deltas{i + 1}) .* activations{i} .* (1 - activations{i}); % Sigmoid gradient
            end
    
            % Update weights and biases
            for i = 1:length(weights)
                weights{i} = weights{i} - alfa * (deltas{i} * activations{i}') / n;
                biases{i} = biases{i} - alfa * sum(deltas{i}, 2) / n;
            end
        
        end
        % Display the loss for monitoring training progress
        fprintf("Epoch[%d]: - EmpiricalRisk: %f\n", epoch, mean(losses));
        if mean(losses) < 0.2
            break
        end
            
    end
end

