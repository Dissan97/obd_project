function [empirical_risk, y_pred] = test_data(w, b,activation_functions, x_t, y_t, attempts, fd, threshold)

    n = size(x_t, 1);
    y_pred = zeros(n, 1);
    losses = zeros(n, 1);
    
    
    for trial =1:attempts
        % Generate a random permutation of indices
        permuted_indices = randperm(n);
        
        % Permute x_test and y_test using the generated indices
        x_test_permuted = x_t(permuted_indices, :);
        y_test_permuted = y_t(permuted_indices);   


            for sample=1:n
                x = x_test_permuted(sample, :);
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
                loss = binary_cross_entropy(y_test_permuted(sample), activations{end});
                losses(sample) = loss;
                y_pred(sample) = activations{end};
            end
            % Display training and validation metrics
        
        [accuracy, recall, precision] = calculate_metrics(y_test_permuted, y_pred, threshold);
        fprintf(fd, "Trial[%d] with threshold=%.4f:\n" + ...
            "precision: %f - recall: %f - accuracy: %f\n" + ...
            "EmpiricalRisk: %f\n",  trial, threshold, accuracy, recall, precision, mean(losses));
    
    end

    empirical_risk = mean(losses);

end

