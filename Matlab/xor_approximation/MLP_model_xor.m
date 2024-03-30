clear;
clc;
close all;

%% Neural Network Arquitecture
n_inputs = 2;  % Input Neurons
n_hiddenNeurons = 3;  % hidden layer neurons
n_outputNeurons = 1;  % Output Neurons

% Random weight initialization
weights1 = randn(n_inputs +1, n_hiddenNeurons); % +1 because of bias
weights2 = randn(n_hiddenNeurons+1, n_outputNeurons); % +1 because of bias

% Trainig Data
originalInputs = [0 0; 0 1; 1 0; 1 1];
desiredOutputs = [0; 1; 1; 0];

% Trainig parameters
learning_rate = 0.1;
epochs = 10000;

% Error Measurement
mse_history = zeros(epochs,1);

%% Training
for epoch = 1:epochs
    squaredErrors = zeros(size(originalInputs,1),1); % Store squared errors for each example

    % for loop for all the inputs (4)

    for i = 1:size(originalInputs,1)
       % Fordward Propagation
       inputVector = [originalInputs(i,:) 1] % adding bias at the end
       HLActivationPotential = input * weights1;             
       HLActivationFunction = sigmoid(HLActivationPotential);
       HLOutput = [HLActivationFunction 1];
       output = sigmoid(HLOutput * weights2);

       % error
       error = desiredOutputs - output;

       OLGradient = error .* sigmoid_derivative(output)


    end
end


%% Functions

% Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Derivada de la función sigmoide
function y = sigmoid_derivative(x)
    y = x .* (1 - x);
end