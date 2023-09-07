function [x_train, y_train, x_validation, y_validation, x_test, y_test] = get_data(file_name, train_ratio, validation_ratio, target)
data = readtable(file_name);
% Estrai il numero di righe nel tuo DataFrame
num_rows = size(data, 1);

% Genera un vettore di indici casuali che rappresenta l'ordine di mescolamento
shuffled_indices = randperm(num_rows);

% Usa questi indici per mescolare le righe dei dati
shuffled_data = data(shuffled_indices, :);


labels = shuffled_data{:, target};
features = shuffled_data;
features(:,target) = [];


num_samples = size(features, 1);

% Calcola il numero di campioni per ciascun set
num_train_samples = round(train_ratio * num_samples);
num_validation_samples = round(validation_ratio * num_samples);


% Estrai i campioni per ciascun set
x_train = features{1:num_train_samples, :};
y_train= labels(1:num_train_samples);

x_validation = features{num_train_samples+1:num_train_samples+num_validation_samples, :};
y_validation = labels(num_train_samples+1:num_train_samples+num_validation_samples);

x_test = features{num_train_samples+num_validation_samples+1:end, :};
y_test = labels(num_train_samples+num_validation_samples+1:end);
end

