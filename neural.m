clc 
clear

architecture = [87, 128, 64,1];
activation_functions = {'relu', 'relu', 'sigmoid'};

%% data definition
alfa = 0.01;

num_epochs =200;

[x_train, y_train, x_validation, y_validation, x_test, y_test] = get_data('data1.csv', 0.7, 0.1);

%%
% Initialize weights and biases for each layer (random initialization)
weights = cell(1, length(architecture) - 1);
biases = cell(1, length(architecture) - 1);

for i = 1:length(weights)
    % Initialize weights using random values
    weights{i} = randn(architecture(i + 1), architecture(i));
    
    % Initialize biases as zeros
    biases{i} = randn(architecture(i + 1), 1);
end


%% training loop gradient descent
n = size(x_train, 1);
y_pred = zeros(n, 1);
losses = zeros(n, 1);
for epoch=1:num_epochs
    
    for sample=1:n
        x = x_train(sample, :);
        activations = cell(1, length(architecture));
        z_values = cell(1, length(architecture) - 1);
        activations{1} = x';
        % forward pass
        for i = 1:length(architecture) - 1
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

        deltas = cell(1, length(architecture) - 1);
        deltas{end} = activations{end} - y_train(sample);
        for i = length(architecture) - 2:-1:1
            if strcmp(activation_functions{i}, 'relu')
                deltas{i} = (weights{i + 1}' * deltas{i + 1}) .* double(z_values{i} > 0); % ReLU gradient
            elseif strcmp(activation_functions{i}, 'sigmoid')
                deltas{i} = (weights{i + 1}' * deltas{i + 1}) .* activations{i} .* (1 - activations{i}); % Sigmoid gradient
            end
        end

        % Update weights and biases
        for i = 1:length(weights)
            weights{i} = weights{i} - alfa * (deltas{i} * activations{i}') / n;
            biases{i} = biases{i} - alfa * sum(deltas{i}, 2) / n;
        end
    
    end
    % Display the loss for monitoring training progress
    [accuracy, recall, precision] = calculate_metrics(y_train, y_pred);
    fprintf("Epoch[%d]:\n" + ...
        "precision: %f - recall: %f - accuracy: %f\n" + ...
        "Mean loss: %f - EmpiricalRisk: %f\n", epoch, accuracy, recall, precision, mean(loss), sum(losses)/n);
end
%% gradient batch descent
batch_size = 128; % Adjust the batch size as needed
n = size(x_train, 1);
y_pred = zeros(n, 1);
losses = zeros(n, 1);
for epoch = 1:num_epochs
    for batch_start = 1:batch_size:n
        % Initialize gradients accumulators for this batch
        gradient_weights = cell(1, length(architecture));
        gradient_biases = cell(1, length(architecture) - 1);
        
        % Process a batch of samples
        batch_end = min(batch_start + batch_size - 1, n);
        for sample = batch_start:batch_end
            x = x_train(sample, :);
            activations = cell(1, length(architecture));
            z_values = cell(1, length(architecture) - 1);
            activations{1} = x';
            % forward pass
            for i = 1:length(architecture) - 1
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
            losses(sample) = loss;
            % backward pass
    
            deltas = cell(1, length(architecture) - 1);
            deltas{end} = activations{end} - y_train(sample);
            for i = length(architecture) - 2:-1:1
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
                gradient_weights{i} = gradient_weights{i} + deltas{i} * activations{i}';
                gradient_biases{i} = gradient_biases{i} + sum(deltas{i}, 2);
            end
        end
        
        % Update weights and biases after processing the batch
        for i = 1:length(weights)
            weights{i} = weights{i} - alfa * (gradient_weights{i} / batch_size);
            biases{i} = biases{i} - alfa * (gradient_biases{i} / batch_size);
        end

    end
    
    % Display the loss for monitoring training progress
    [accuracy, recall, precision] = calculate_metrics(y_train, y_pred);
    fprintf("Epoch[%d]:\n" + ...
        "precision: %f - recall: %f - accuracy: %f\n" + ...
        "loss: %f - EmpiricalRisk: %f\n", epoch, accuracy, recall, precision, loss, sum(losses)/n);
end

%% batch with validation
   % Initialize variables for storing metrics
lambda = 0.01;
n = size(x_train, 1);
y_pred = zeros(n, 1);
losses = zeros(n, 1);
train_losses = zeros(1, num_epochs);
train_accuracies = zeros(1, num_epochs);
validation_losses = zeros(1, num_epochs);
validation_accuracies = zeros(1, num_epochs);

for epoch = 1:num_epochs
    % Training
    for batch_start = 1:batch_size:length(x_train)
        % Initialize gradients accumulators for this batch
        gradient_weights = cell(1, length(architecture));
        gradient_biases = cell(1, length(architecture) - 1);

        % Process a batch of samples
        batch_end = min(batch_start + batch_size - 1, length(x_train));
        for sample = batch_start:batch_end
            x = x_train(sample, :);
            activations = cell(1, length(architecture));
            z_values = cell(1, length(architecture) - 1);
            activations{1} = x';

            % Forward pass
            for i = 1:length(architecture) - 1
                z_values{i} = weights{i} * activations{i} + biases{i};

                % Activation function
                if strcmp(activation_functions{i}, 'relu')
                    activations{i + 1} = relu(z_values{i});
                elseif strcmp(activation_functions{i}, 'sigmoid')
                    activations{i + 1} = sigmoid(z_values{i});
                end
            end
            y_pred(sample) = activations{end};

            % Calculate loss function with L1 regularization
            loss = binary_cross_entropy(y_train(sample), activations{end});
            regularization_term = 0;  % Initialize regularization term

            for i = 1:length(weights)
                regularization_term = regularization_term + sum(abs(weights{i}(:)));  % L1 penalty term
            end

            loss = loss + (lambda / (2 * length(x_train))) * regularization_term;  % Add L1 regularization term to the loss
            losses(sample) = loss;

            % Backward pass
            deltas = cell(1, length(architecture) - 1);
            deltas{end} = activations{end} - y_train(sample);

            for i = length(architecture) - 2:-1:1
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

            % Accumulate gradients for this sample with L1 regularization
            for i = 1:length(weights)
                gradient_weights{i} = gradient_weights{i} + (deltas{i} * activations{i}') + (lambda / length(x_train)) * sign(weights{i});  % Add L1 penalty gradient
                gradient_biases{i} = gradient_biases{i} + sum(deltas{i}, 2);
            end
        end

        % Update weights and biases after processing the batch
        for i = 1:length(weights)
            weights{i} = weights{i} - alfa * (gradient_weights{i} / batch_size);
            biases{i} = biases{i} - alfa * (gradient_biases{i} / batch_size);
        end
    end

    % Calculate training metrics for this epoch
    [accuracy, ~, ~] = calculate_metrics(y_train, y_pred);
    train_losses(epoch) = sum(losses) / length(x_train);
    train_accuracies(epoch) = accuracy;

    % Validation
    validation_losses(epoch) = 0;
    validation_accuracies(epoch) = 0;
    for sample = 1:length(x_validation)
        x = x_validation(sample, :);
        activations = cell(1, length(architecture));
        activations{1} = x';

        % Forward pass
        for i = 1:length(architecture) - 1
            z_values{i} = weights{i} * activations{i} + biases{i};

            % Activation function
            if strcmp(activation_functions{i}, 'relu')
                activations{i + 1} = relu(z_values{i});
            elseif strcmp(activation_functions{i}, 'sigmoid')
                activations{i + 1} = sigmoid(z_values{i});
            end
        end

        % Calculate loss function for validation data
        validation_loss = binary_cross_entropy(y_validation(sample), activations{end});
        validation_losses(epoch) = validation_losses(epoch) + validation_loss;

        % Calculate accuracy for validation data
        y_val_pred = activations{end};
        validation_accuracies(epoch) = validation_accuracies(epoch) + (y_validation(sample) == (y_val_pred >= 0.5));
    end
    validation_losses(epoch) = validation_losses(epoch) / length(x_validation);
    validation_accuracies(epoch) = validation_accuracies(epoch) / length(x_validation);

    % Display training and validation metrics
    [accuracy, recall, precision] = calculate_metrics(y_train, y_pred);
    fprintf("Epoch[%d]:\n" + ...
        "precision: %f - recall: %f - accuracy: %f\n" + ...
        "loss: %f - EmpiricalRisk: %f\n", epoch, accuracy, recall, precision, loss, sum(losses)/n);
end

%% test

n = size(x_test, 1);
y_pred = zeros(n, 1);
losses = zeros(n, 1);

num_test_samples = n;
for trial =1:10
    % Generate a random permutation of indices
    permuted_indices = randperm(num_test_samples);
    
    % Permute x_test and y_test using the generated indices
    x_test_permuted = x_test(permuted_indices, :);
    y_test_permuted = y_test(permuted_indices);
    for epoch=1:num_epochs
    
        for sample=1:n
            x = x_test_permuted(sample, :);
            activations = cell(1, length(architecture));
            z_values = cell(1, length(architecture) - 1);
            activations{1} = x';
            % forward pass
            for i = 1:length(architecture) - 1
                z_values{i} = weights{i} * activations{i} + biases{i};
    
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
    end
        % Display training and validation metrics
    [accuracy, recall, precision] = calculate_metrics(y_test_permuted, y_pred);
    fprintf("Trial[%d]:\n" + ...
        "precision: %f - recall: %f - accuracy: %f\n" + ...
        "loss: %f - EmpiricalRisk: %f\n", trial, accuracy, recall, precision, loss, sum(losses)/n);
end
