function [weights, biases, epoch] = gradient_descent_batch(x_train, y_train, activation_functions,weights, biases, num_epochs, alfa, batch_size, lambda)
    n = size(x_train, 1);
    y_pred = zeros(n, 1);
    losses = zeros(n, 1);
    length_architecture = length(weights) + 1;

    regularize = 1;
    if lambda <= 0
        regularize = 0;
    end


    for epoch = 1:num_epochs
        for batch_start = 1:batch_size:n
            % Initialize gradients accumulators for this batch
            gradient_weights = cell(1, length_architecture);
            gradient_biases = cell(1, length_architecture - 1);
            
            % Process a batch of samples
            batch_end = min(batch_start + batch_size - 1, n);
            for sample = batch_start:batch_end
                x = x_train(sample, :);
                activations = cell(1, length_architecture);
                z_values = cell(1, length_architecture - 1);
                activations{1} = x';
                % forward pass
                for i = 1:length_architecture - 1
                    z_values{i} = weights{i} * activations{i} + biases{i};
        
                    % activation function
                    if strcmp(activation_functions{i}, 'relu')
                        activations{i + 1} = relu(z_values{i});
                    elseif strcmp(activation_functions{i}, 'sigmoid')
                        activations{i + 1} = sigmoid(z_values{i});
                    end
                end
                y_pred(sample) = activations{end};
                % calculate loss function 
                loss = binary_cross_entropy(y_train(sample), activations{end});
                
                
                if regularize == 1   
                    regularization_term = 0;  % Initialize regularization term

                    for i = 1:length(weights)
                    regularization_term = regularization_term + sum(abs(weights{i}(:)));  % L1 penalty term
                    end

                    loss = loss + (lambda / (2 * length(x_train))) * regularization_term;  % Add L1 regularization term to the loss
                end

                losses(sample) = loss;

                % backward pass
        
                deltas = cell(1, length_architecture - 1);
                deltas{end} = activations{end} - y_train(sample);
                for i = length_architecture - 2:-1:1
                    if strcmp(activation_functions{i}, 'relu')
                        deltas{i} = (weights{i + 1}' * deltas{i + 1}) .* double(z_values{i} > 0); % ReLU gradient
                    elseif strcmp(activation_functions{i}, 'sigmoid')
                        deltas{i} = (weights{i + 1}' * deltas{i + 1}) .* activations{i} .* (1 - activations{i}); % Sigmoid gradient
                    end
                end
                 % Initialize gradients for this sample
                if sample == batch_start
                    for i = 1:length(weights)
                        gradient_weights{i} = zeros(size(weights{i}));
                        gradient_biases{i} = zeros(size(biases{i}));
                    end
                end
                
                % Accumulate gradients for this sample
                for i = 1:length(weights)
                    gradient_weights{i} = gradient_weights{i} + deltas{i} * activations{i}' + regularize * (lambda / length(x_train)) * sign(weights{i});  % Add L1 penalty gradient;
                    gradient_biases{i} = gradient_biases{i} + sum(deltas{i}, 2);
                end
            end
            
            % Update weights and biases after processing the batch
            for i = 1:length(weights)
                weights{i} = weights{i} - alfa * (gradient_weights{i} / batch_size);
                biases{i} = biases{i} - alfa * (gradient_biases{i} / batch_size);
            end
    
        end
        if mean(losses) < 0.02
            break
        end
        fprintf("Epoch[%d]: EmpiricalRisk: %f\n", epoch, mean(losses));
    end
end

