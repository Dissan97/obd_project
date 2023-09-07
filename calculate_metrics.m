function [accuracy, recall, precision] = calculate_metrics(y_true, y_pred, th)


    % Convert probabilities to binary predictions (0 or 1)
    y_pred_binary = (y_pred >= th);

    % Calculate accuracy
    accuracy = sum(y_true == y_pred_binary) / numel(y_true);

    % Calculate true positives, false positives, true negatives, false negatives
    true_positives = sum(y_true == 1 & y_pred_binary == 1);
    false_positives = sum(y_true == 0 & y_pred_binary == 1);
    false_negatives = sum(y_true == 1 & y_pred_binary == 0);

    % Calculate recall and precision
    recall = true_positives / (true_positives + false_negatives);
    precision = true_positives / (true_positives + false_positives);
end
