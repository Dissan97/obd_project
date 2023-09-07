function [weights,biases] = initialize_architecture(architecture)
    % Initialize weights and biases for each layer (random initialization)
    weights = cell(1, length(architecture) - 1);
    biases = cell(1, length(architecture) - 1);
    
    for i = 1:length(weights)
        % Initialize weights using random values
        weights{i} = randn(architecture(i + 1), architecture(i));
        
        % Initialize biases as zeros
        biases{i} = randn(architecture(i + 1), 1);
    end
end

