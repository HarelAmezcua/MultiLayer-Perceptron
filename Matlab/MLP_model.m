clear;
close all;
clc;

[pc_1, pc_2] = simple_mlp()


%% Functions
function [pc1, pc2] = simple_mlp()
    % Network Parameters
    n_inputVector = 2;    % Number of inputs
    n_hiddenNeurons = 3;   % Number of neurons on the hidden layer
    n_outputNeurons = 1;   % Number of neurons on the output layer

    % Random weight initialization
    weights1 = randn(n_inputVector + 1, n_hiddenNeurons)    % +1 para el sesgo
    weights2 = randn(n_hiddenNeurons + 1, n_outputNeurons)   % +1 para el sesgo

    % Datos de entrenamiento (XOR)
    originalInputs = [0 0; 0 1; 1 0; 1 1];
    Y = [0; 1; 1; 0];

    % Parámetros de entrenamiento
    learning_rate = 0.1;
    epochs = 1000000;

   % Initialize MSE history
    mse_history = zeros(epochs, 1);

    % Entrenamiento
    for epoch = 1:epochs
        squaredErrors = zeros(size(originalInputs, 1), 1); % Store squared errors for each example
        % Bucle sobre todos los ejemplos
        for i = 1:size(originalInputs, 1)
            % Propagación hacia adelante
            input = [originalInputs(i, :) 1]; % Agregar sesgo
            hidden = sigmoid(input * weights1);
            hidden = [hidden 1]; % Agregar sesgo
            output = sigmoid(hidden * weights2);

            % Error
            error = Y(i) - output;

            % Retropropagación del error y actualización de pesos
            dW2 = error .* sigmoid_derivative(output) .* hidden';
            weights2 = weights2 + learning_rate * dW2;

            dW1 = (error .* sigmoid_derivative(output)) * weights2(1:end-1)' .* sigmoid_derivative(hidden(1:end-1)) .* input';
            weights1 = weights1 + learning_rate * dW1;

            % Store squared error for this example
            squaredErrors(i) = error^2;           
        end
        % Compute MSE for this epoch and store it
        mse_history(epoch) = mean(squaredErrors);
    end  
    pc1 = weights1;
    pc2 = weights2;

    plot(mse_history)
end



% Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Derivada de la función sigmoide
function y = sigmoid_derivative(x)
    y = x .* (1 - x);
end