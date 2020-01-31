% ================================================================
%                   Introduction to Machine Learning  
%
%                             Lab 3
% 
% =================================================================
% Instructions:
% Follow the code and the comments in this file carefully. You will need to 
% change parts of the code in this file and in other files. The places
% where you need to enter your code is indicated by:
% ====================== YOUR CODE HERE ======================
% ...
% ============================================================
%
% A written report should be handed no later than one week after the lab.
% The report is graded with either PASS or PASS WITH DISTINCTION. In each 
% cell of code (a cell is separated by %%) there are instructions for what 
% to include in the report, e.g.:
% ====================== REPORT ==============================
% Some instructions...
% FOR PASS WITH DISTINCTION: Some more instructions...
% ============================================================
% All the instructions for extra credits have to be 
% completed in order to receive a PASS WITH DISTINCTION on the report.
%
% It is recommended to complete all the YOUR CODE HERE blocks before 
% working on the REPORT blocks.
% 
% MATLAB Tips:
% You can read about any MATLAB function by using the help function, e.g.:
% >> help plot
% To run a single line of code simply highlight the code you want to run 
% and press F9. To run one cell of code first click inside the cell and 
% then press CTRL + ENTER.

%% ===================== Preparation ===========================
% Download lab3.zip from Blackboard and unzip it to  
% somewhere on your computer. Change the path using the command cd to the
% folder where your files are located.
% ====================== YOUR CODE HERE ======================
cd('...')
% ============================================================

addpath('./minFunc/'); %add minFunc to working directory after running the cd command 

%% ========= Part 1a: Regularization for Logistic Regression =============
% First we will add regularization to our previously written logistic
% regression. Copy costLogisticRegression.m, sigmoid.m, and checkGradient.m
% from lab 1 and implement L2 weight decay regularization in 
% costLogisticRegression.m. 
clear all;

% We will test if regularization can help an email spam detector. 
load spamTrain
X = [ones(size(X,1), 1) X]; % add intercept term

% The test set is located in spamTest.mat. We need to divide the
% training data into train and validation sets. 
[Xtrain, Xval] = splitData(X, [0.8; 0.2], 0);
[ytrain, yval] = splitData(y, [0.8; 0.2], 0);
initial_theta = zeros(size(X, 2), 1);
clear X

% Implement L2 weight decay regularization in costLogisticRegression.m. 
% Do not regularize the first element in theta. Check the gradients on a
% small subset of the data.
lambda = 1e-4;
[J, grad] = costLogisticRegression(initial_theta, Xtrain(1:10,:), yval(1:10), lambda);
numgrad = checkGradient(@(p) costLogisticRegression(p, Xtrain(1:10,:), yval(1:10), lambda), initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Train a logistic regression model on the full train set. We have
% provided two choices for optimization solver; minFunc and fmincg. You can 
% also use Matlab's fminunc or write your own similar to gradientDescent.m 
% from lab 1.
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
lambda = 1e-4;
theta = fmincg(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);
%theta = minFunc(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);

% Calculate the classification accuracy on the train and validation set
trainaccuracy = mean(round(sigmoid(Xtrain*theta))==ytrain)
valaccuracy = mean(round(sigmoid(Xval*theta))==yval)

% The classification error is calculated as the 1-accuracy.
trainerror = 1 - trainaccuracy
valerror = 1 - valaccuracy

% ====================== YOUR CODE HERE ======================
% Make a plot of the classification error on the train and validation sets
% as a function of lambda. Try the following values for lambda:
% lambda_list = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1]. You can use
% set(gca,'Xtick', 1:length(lambda_list), 'Xticklabel', lambda_list)
% to set the x-label as the values for lambda


% ============================================================

% ====================== REPORT ==============================
% Show the plot of the train and validation error as a function of lambda. 
% What is the best choice for lambda? What is the classification accuracy 
% on the test set (type load spamTest to load the test set. Don't forget to 
% add intercept term) with this chosen lambda? What would the classification 
% accuracy on the test set be if lambda = 0?
% ============================================================


%% ================= Part 1b: n-fold cross-validation =====================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.


% ====================== YOUR CODE HERE ======================
% Perform a 5-fold cross-validation on the training data with the optimal 
% choice of lambda from Part 1a. 
% ============================================================

% ====================== REPORT ==============================
% FOR PASS WITH DISTINCTION: Show the code and report the mean and standard 
% deviation of a 5-fold cross-validation.
% =============================================================

%% ========= Part 2a: Logistic Regression for multiple classes =============
% In this part we will train a logistic regression classifier for the task
% of classifying handwritten digits [0-9]. 
clear all;

% First we load the data from the file smallMNIST.mat which is a reduced 
% set of the MNIST handwritten digit dataset. The full data set can be
% downloaded from http://yann.lecun.com/exdb/mnist/. Our data X consist of 
% 5000 examples of 20x20 images of digits between 0 and 9. The number "0" 
% has the label 10 in the label vector y. The data is already normalized.
load('smallMNIST.mat'); % Gives X, y

% We use displayData to view 100 random examples at once. 
[m, n] = size(X);
rand_indices = randperm(m);
figure; displayData(X(rand_indices(1:100), :));

% Now we divide the data X and label vector y into training, validation and
% test set. We use the same seed so that we dont get different
% randomizations. We will use hold-out cross validation to select the
% hyperparameter lambda.
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6; 0.3; 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6; 0.3; 0.1], seed);

% Now we train 10 different one vs all logistic regressors. Complete the
% code in trainLogisticReg.m before continuing. 
lambda = 0.01;
all_theta = trainLogisticReg(Xtrain, ytrain, lambda);

% Now we calculate the predictions using all 10 models. 
ypredtrain = predictLogisticReg(all_theta, Xtrain);
ypredval = predictLogisticReg(all_theta, Xval);
ypredtest = predictLogisticReg(all_theta, Xtest);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

% It could be interesting to plot the missclassified examples.
% figure; displayData(Xtest(ypredtest~=ytest, :));

% ====================== REPORT ==============================
% Report what classification accuracy you got on the train, test, and validation set.
% ============================================================


%% ========= Part 2b: Softmax classification for multiple classes ==========
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% In this part we will train a softmax classifier for the task of 
% classifying handwritten digits [0-9]. 
clear  all;

% Load the same data set. In softmax and neural networks the convention is 
% to let each column be one training input instead of each row as we have 
% previously used. 
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';

% Split into train, val, and test sets
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], seed);

% Initialize theta
numClasses = 10; % Number of classes
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);

% For debugging purposes create a small randomized data matrix and
% labelvector. Calculate cost and grad and check gradients. Finish the code 
% in costSoftmax.m first. If your gradients don't match at first, try setting 
% lambda = 0; to see if the problem is with the error term or the 
% regularization term.
lambda = 1e-4;
[cost,grad] = costSoftmax(initial_theta, Xtrain(:,1:10), ytrain(1:10), numClasses, lambda);
numGrad = checkGradient( @(p) costSoftmax(p, Xtrain(:,1:10), ytrain(1:10), numClasses, lambda), initial_theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

% Now we train the softmax classifier.
lambda = 0.01;
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
theta = trainSoftmax(Xtrain, ytrain, numClasses, lambda, options);

% Now we calculate the predictions.
ypredtrain = predictSoftmax(theta, Xtrain, numClasses);
ypredval = predictSoftmax(theta, Xval, numClasses);
ypredtest = predictSoftmax(theta, Xtest, numClasses);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

% ====================== REPORT ==============================
% PASS WITH DISTINCTION: Show the code in costSoftmax.m.
% What accuracy on the train, val, and test set did you get?
% ============================================================


%% ============Part 2c: Plot Learning curve =====================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% ====================== REPORT ==============================
% PASS WITH DISTINCTION: Load the smallMNIST.mat data set and choose either 
% logistic regression or softmax classifier and make a learning curve 
% analysis in order to find out if we could get a better result if we had more 
% training data. Show a plot of the learning curve and discuss if it would 
% be worth to get more training data. 
% ============================================================

%% ============== Part 3a: Implementing Neural network ====================
% Time to implement a neural network. The sparsity regularization is only
% needed to be implemented for PASS WITH DISTINCTION.
clear  all;

% Create a small randomized data matrix and labelvector for testing your 
% implementation.
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. We start with coding the NN without any
% regularization
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter
parameters.beta = 0; % sparsity penalty parameter (For PASS WITH DISTINCTION)

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 10 output units. 
[theta thetaSize] = initNNParameters(8, 5, 10);

% Calculate cost and grad and check gradients. Finish the code in 
% costNeuralNetwork.m first.
[cost,grad] = costNeuralNetwork(theta, thetaSize, X, y, parameters);
numGrad = checkGradient( @(p) costNeuralNetwork(p, thetaSize, X, y, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

% ====================== REPORT ===========================================
% Show the code in costNeuralNetwork.m.
% FOR PASS WITH DISTINCTION: Also implement the sparsity regularization
% =========================================================================

%% ==== Part 3b: Neural network for handwritten digit classification ======
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

clear all;

% Load the data set and split into train, val, and test sets.
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter
parameters.beta = 0; % This is a tunable hyperparameter
numhid = 50; % % This is a tunable hyperparameter

% Initiliaze the network parameters.
numvis = size(X, 1);
numout = length(unique(y));
[theta, thetaSize] = initNNParameters(numvis, numhid, numout);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costNeuralNetwork(p, thetaSize, Xtrain, ytrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
[optTheta, optCost] = minFunc(costFunction, theta, options);
toc

% fmincg takes longer to train. Uncomment if you want to try it.
% tic
% options = optimset('MaxIter', 400, 'display', 'on');
% [optTheta, optCost] = fmincg(costFunction, theta, options);
% toc

% You can visualize what the network has learned by plotting the weights of
% W1 using displayData.
[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);
displayData(W1);

% Now we predict all three sets.
ypredtrain = predictNeuralNetwork(optTheta, thetaSize, Xtrain);
ypredval = predictNeuralNetwork(optTheta, thetaSize, Xval);
ypredtest = predictNeuralNetwork(optTheta, thetaSize, Xtest);

fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100);
fprintf('Val Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

% ====================== REPORT ==============================
% PASS WITH DISTINCTION: Perform a bias-variance analysis on numhid, lambda,
% and beta to decide the optimal values. Show the 3 bias-variance analysis 
% plots and describe what are the best choices for the 3 hyperparameters.
% ============================================================

%% ============== Part 4a: Implementing Auto-encoder =======================
% Time to implement an auto-encoder.
clear  all;

% Create a small randomized data matrix and labelvector
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter
parameters.beta = 0; % sparsity penalty parameter

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 8 output units (same as the number of input
% units).
[theta, thetaSize] = initAEParameters(8, 5);

% Calculate cost and grad and check gradients. Note how costAutoencoder.m 
% does not require the label vector y.
[cost,grad] = costAutoencoder(theta, thetaSize, X, parameters);
numGrad = checkGradient( @(p) costAutoencoder(p, thetaSize, X, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

% ====================== REPORT ==============================
% Show the code in costAutoencoder.m and report what diff you got.
% ============================================================

%% ======= Part 4b: Reconstructing with Auto-encoder ===================

clear all;

load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % This is a tunable hyperparameter
parameters.beta = 0; % This is a tunable hyperparameter
numhid = 50; % % This is a tunable hyperparameter

% Initiliaze the network parameters. Here we use initAEParameters.m
% instead.
numvis = size(X, 1);
[theta, thetaSize] = initAEParameters(numvis, numhid);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costAutoencoder(p, thetaSize, Xtrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
[optTheta, optCost] = minFunc(costFunction, theta, options);
toc

% fmincg takes longer to train. Uncomment if you want to try it.
% tic
% options = optimset('MaxIter', 400, 'display', 'on');
% [optTheta, optCost] = fmincg(costFunction, theta, options);
% toc

% You can visualize what the network has learned by plotting the weights of
% W1 using displayData.
[W1, W2, b1, b2] = theta2params(optTheta, thetaSize);
displayData(W1);

figure;
h = sigmoid(bsxfun(@plus, W1*Xtrain, b1)); %hidden layer
Xrec = sigmoid(bsxfun(@plus, W2*h, b2)); % reconstruction layer
subplot(1,2,1); displayData(Xtrain(:,1:100)'); title('Original input')
subplot(1,2,2); displayData(Xrec(:,1:100)'); title('Reconstructions')

figure; 
imagesc(h); title(['Mean hidden unit activation: ' num2str(mean(mean(h,2)))])

% ====================== REPORT ===========================================
% Select good parameters so that the reconstructions look like the original
% input data. Show the plot of the original input and the reconstructions. 
% =========================================================================

%% ======= Part 4c: Classification with Auto-encoder ===================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);
numClasses = 10;

% ====================== YOUR CODE HERE ======================
% Use the trained auto-encoder from part 4b and use
% a classifier of your choice (use trainLogisticReg or trainSoftmax) and
% calculate the classification accuracy on the test set.

% ============================================================

% ====================== REPORT ===========================================
% PASS WITH DISTINCTION:  Report the result
% you got and show the code how you calculated the accuracy.
% =========================================================================

