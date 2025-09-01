# ADAS-MLP-demo
Minimal demo code for ADAS deep MLP with self-attention in paper

%% Minimal Public Demo: Deep MLP + Self-Attention for 11->2 Regression
% Purpose: show the network architecture and training flow without real data.
% Notes:
% - Replace synthetic data with your own (same dimensions) in private code.
% - Keep hyperparameters at your discretion; here we show sensible values.
% - If selfAttentionLayer is not available, set USE_ATTENTION = false.

clc; clear; close all;

%% 0) Synthetic data (placeholder, NOT real)
% Feature dimension: 11, Target dimension: 2
rng(42);                      % reproducibility
N      = 2000;               % total samples
Dx     = 11;                 % #features (match your private pipeline)
Dy     = 2;                  % #outputs  (ΔGf, ΔGr-like)

X      = randn(Dx, N);       % synthetic features
Y      = [0.7  -0.3]'.*X(1:2,:) + 0.1*randn(Dy, N);  % synthetic targets (dummy relation)

%% 1) Train/Test split (80/20)
idx       = randperm(N);
nTrain    = floor(0.8 * N);
train_idx = idx(1:nTrain);
test_idx  = idx(nTrain+1:end);

Xtr = X(:, train_idx);
Ytr = Y(:, train_idx);
Xte = X(:, test_idx);
Yte = Y(:, test_idx);

%% 2) Normalization (min-max)
[Xtr, PSx] = mapminmax(Xtr);
[Ytr, PSy] = mapminmax(Ytr);
Xte = mapminmax('apply', Xte, PSx);

%% 3) Network definition (keeps your architecture shape)
USE_ATTENTION = true;  % set false if selfAttentionLayer is unavailable

layers = [
    featureInputLayer(Dx, 'Name', 'input')                           % input: 11
    fullyConnectedLayer(500, 'Name', 'fc1', 'WeightL2Factor', 1e-4)  % FC1
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(100, 'Name', 'fc2', 'WeightL2Factor', 1e-4)  % FC2
    reluLayer('Name', 'relu2')
];

% expand + self-attention block (optional)
if USE_ATTENTION
    layers = [
        layers
        fullyConnectedLayer(100, 'Name', 'expand', 'WeightL2Factor', 1e-4)
        selfAttentionLayer(10, 100, "NumValueChannels", "auto")      % multi-head style
    ];
end

% tail MLP + regression head
layers = [
    layers
    fullyConnectedLayer(100, 'Name', 'fc3', 'WeightL2Factor', 1e-4)
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(100, 'Name', 'fc4', 'WeightL2Factor', 1e-4)
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(100, 'Name', 'fc5', 'WeightL2Factor', 1e-4)
    reluLayer('Name', 'relu5')
    fullyConnectedLayer(Dy,  'Name', 'output')                       % output: 2
    regressionLayer('Name', 'regression')
];

%% 4) Training options (concise but robust)
options = trainingOptions('adam', ...
    'MaxEpochs',         800, ...
    'InitialLearnRate',  1e-4, ...
    'MiniBatchSize',     256, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 300, ...
    'LearnRateDropFactor', 0.9, ...
    'Shuffle',           'every-epoch', ...
    'Verbose',           false);

%% 5) Train
net = trainNetwork(Xtr', Ytr', layers, options);  % samples on rows

%% 6) Predict & inverse transform
Yhat = predict(net, Xte')';
Yhat = mapminmax('reverse', Yhat, PSy);

%% 7) Evaluate (RMSE per output)
rmse = sqrt(mean((Yhat - Yte).^2, 2));
fprintf('RMSE per output: [%.4f, %.4f]\n', rmse(1), rmse(2));
