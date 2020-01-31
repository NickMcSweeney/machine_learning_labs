% ================================================================
%                   Introduction to Machine Learning
%
%                             Lab 1
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
% The report is graded with either PASS or PASS WITH DISTINCTION. At the
% end of each cell of code (a cell is separated by %%) there are
% instructions for what to include in the report, e.g.:
% ====================== REPORT ==============================
% Some instructions...
% FOR PASS WITH DISTINCTION: Some more instructions...
% ============================================================
% All the report blocks that contain FOR PASS WITH DISTINCTION need
% to be completed in order to receive a PASS WITH DISTINCTION on the report.
%
% MATLAB Tips:
% You can read about any MATLAB function by using the help function, e.g.:
% >> help plot
% To run a single line of code simply highlight the code you want to run
% and press F9. To run one cell of code first click inside the cell and
% then press CTRL + ENTER.

%% ===================== Preparation ===========================
% Download lab1.zip from Blackboard and unzip it to
% somewhere on your computer. Change the path using the command cd to the
% folder where your files are located.
% ====================== YOUR CODE HERE ======================
%cd(...
% ============================================================

%% ======================= Part 1a: Plotting =======================
load carbig.mat
% This command loads a number of variables to the MATLAB workspace. The
% variables contain information about cars such as WEIGHT, Horsepower,
% Model_Year e.t.c. The task is to fit a linear model that predicts the
% horsepower based on the weight of the car.

x = Weight; % data vector
y = Horsepower; % prediction values

% The first thing to do when working with a new data set is to plot it. For
% this data set you don't want to draw any lines between each data point so
% set the plot symbol to point "." Type ">> help plot" to see how.
% ====================== YOUR CODE HERE ======================
%plot(x,y,"r.");
% ============================================================

%xlabel('Weight [kg]');
%ylabel('Horsepower [hp]');

% ====================== REPORT ==============================
% Present the problem and describe the data. Show a plot of the data.
%
% In this lab we have two relational data vectors, weight and horsepower, given the initial belief that there is a relationship between these two data elements we think that if we know the weight of the car, we will be able to predict the horsepower. To do this we will fit a linear model to the car data as the point cloud we plotted suggest that the data is roughly linear.
% ============================================================

%% ======================= Part 1b: Remove NaN values ==============
% Next we need to clean up the data and remove any training data that
% contains any NaN (not-a-number) or Inf (infinity) values.

% Complete the code in RemoveData.m
[x y] = RemoveData(x, y);

%% ======================= Part 1c: Normalize data ==============
% Scale the features down using mean normalization.
% Complete the code in featureMeanNomralize.m
[x mu sigma] = featureMeanNormalize(x);
%figure;
%plot(x,y,"r.");

% ====================== REPORT ==============================
% Explain why it is necessary to normalize the data.
% Hint: Try to run the gradient descent algorithm without normalizing
% the data and see what happends.
%
% Without normalization we are unable to get a proper regression of the data. The normalization gives the data a zero mean, moving the data to be centered around zero, rather than an arbitrary point, this normalization, while not required, gives the data a similar range, speeding up the gradient decent. As such in our timeframe the un-normalized data fails in gradient decent.
% ============================================================

%% ======= Part 2: Linear Regression: Cost function and derivative ========
X = [ones(size(x,1), 1), x]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% compute cost and derivative.
% Complete the code in costLinearRegression.m before continuing.
[J grad] = costLinearRegression(theta, X, y);

% You can check if your solution is correct by calculating the numerical
% gradients (numgrad) and comhelppare them with the analytical gradients
% (grad). Complete the code in checkGradient.m before continuing.
numgrad = checkGradient(@(p) costLinearRegression(p, X, y), theta);

% If your implementation is correct the two columns should be very similar.
disp([numgrad grad]);
% and the diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad)
disp("my diff")
% ====================== REPORT ==============================
% What value of diff did you get?
% diff: 2.393e-10
% ============================================================

%% =================== Part 3: Gradient descent ===================
X = [ones(size(x,1), 1), x]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Hyperparameters for gradient descent
num_iters = 1500;
alpha = 0.01;

% Run Gradient Descent. Complete the code in gradientDescent.m.
[theta_mo J_history_mo] = gradientDescentMomentum(theta, X, y, alpha, num_iters);
[theta J_history] = gradientDescent(theta, X, y, alpha, num_iters);

% Plot J_history. If your implementation is correct J should decrease after
% each iteration.
figure;
plot(J_history, 'k-');
hold on;
plot(J_history_mo, 'g-');
legend('Standard','With Momentum')
xlabel('Iteration')
ylabel('Cost J(\theta)')

% Plot the data and the linear regression model
figure;
plot(X(:,2), y, 'b.'); hold on;
plot(X(:,2), X*theta, 'r-')
legend('Training data', 'Linear regression')

% ====================== REPORT ==============================
% Include the plot of J_history and the plot of the linear fit. Discuss the
% plots. Is the number of iterations in the gradient descent sufficient to
% find a good solution? How would a different value of alpha influence the
% found optimal values of theta and the speed of convergence?
% FOR PASS WITH DISTINCTION: Implement momentum and show a plot of
% J_history with and without momentum in the same plot with different
% colors. Include a legend.
%
%
% ============================================================

%% ======= Part 4a: Linear regression with multiple variables ==============
% In this part you are going to change the code in the following files so
% that your implementation of Linear Regression works for multiple
% variables:
%       RemoveData.m
%       featureMeanNormalize.m
%       costLinearRegression.m

% Load data
x = [Weight MPG]; % We use two variable - weight and miles per gallon (MPG)
y = Horsepower; % prediction values

% Plot the data. Use Tools -> Rotate 3D to examine the data.
figure;
plot3(x(:,1),x(:,2),y,'.','Color','b')
xlabel('Weight [kg]');
ylabel('Fuel efficiency [MPG]');
zlabel('Horsepower [hp]');
grid on

% Remove pairs of data that contains any NaN values. Change the code in
% RemoveData.m so that it works for multiple variables.
[x y] = RemoveData(x, y);

% Normalize both feature vectors
[x mu sigma] = featureMeanNormalize(x);

% ====================== REPORT ==============================
% Include the final code in RemoveData.m and featureMeanNormalize.m. How
% many training examples did you have left after removing some of the data?
% What was mu and sigma?
% ============================================================

X = [ones(size(x,1), 1) x]; % Add intercept term to X
theta = zeros(3, 1);

% Check gradients. You might need to change costLinearRegression.m so that
% it works for multiple variables.
[J grad] = costLinearRegression(theta, X, y);
numgrad = checkGradient(@(p) costLinearRegression(p, X, y), theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Hyperparameters for gradient descent
alpha = 0.1;
num_iters = 500;

% Run Gradient Descent
[theta J_history] = gradientDescent(theta, X, y, alpha, num_iters);

% Plot J_history. If your implementation is correct J should decrease after
% each iteration.
plot(J_history)

% Plot the data and the linear regression model
plot3(X(:,2), X(:,3), y, 'b.'); hold on;
range=-2:.1:2;
[xind,yind] = meshgrid(range);
Z = zeros(size(xind));
for i=1:size(xind,1)
    for j=1:size(xind,2)
        Z(i,j) = [1 xind(i,j) yind(i,j)]*theta;
    end
end
surf(xind,yind,Z)
shading flat; grid on;
xlabel('Normalized Weight');
ylabel('Normalized MPG');
zlabel('Horsepower [hp]')
legend('Training data', 'Linear regression')

% Predict how much horsepower a car would have that weights 3000 kg and has
% a MPG of 30. Hint: It should be around 98
% ====================== YOUR CODE HERE ======================
% Remember that you have to normalize the values first using mu and sigma.
%y_pred = ...
% ============================================================

% ====================== REPORT ==============================
% Show the final code in costLinearRegression.m. What value of diff did you
% get? What alpha and num_iters did you use? What values of theta did you
% get? Show the plot of J_history. What prediction of the horsepower did
% you get? Show how you calculated that value.
% ============================================================

%% ========= Part 4b: Vectorization - Linear regression with multiple variables ==================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% ====================== REPORT ==============================
% FOR PASS WITH DISTINCTION: Vectorize the code in featureMeanNormalize.m,
% RemoveData.m, and costLinearRegression.m so that there are no for-loops.
% ============================================================


%% ================ Part 5: Normal Equation ================

x = [Weight MPG]; % We use two variable - weight and miles per gallon (MPG)
y = Horsepower; % prediction values
[x y] = RemoveData(x, y); % Remove bad training data
X = [ones(size(x,1), 1) x]; % Add intercept term to X

% Calculate the parameters from the normal equation. You need to finish the
% code in normalEqn first.
theta = normalEqn(X, y);

% Predict how much horsepower a car would have that weights 3000 kg and has
% a MPG of 30. You should get the same answer as in Part 4.
% ====================== YOUR CODE HERE ======================
%y_pred_normalEqn = ...
% ============================================================

% ====================== REPORT ==============================
% Show the final code in normalEqn.m. Compare the predicted horsepower when
% using the normal equation and when using gradient descent. Show how you
% calculated that value.
% ============================================================


%% ==================== Part 6: Logistic Regression ====================
clear; % Clear all workspace variables
load hospital.mat
% This data set contain the information about age, sex, weight, and blood
% pressure for 100 patients. Your task is to train a logistic regression
% classifier to classify whether a patient is a smoker or a non-smoker.

x = [hospital.Age hospital.BloodPressure(:,1)]; %We start with two input
% features - age and blood pressure
y = hospital.Smoker; % Label vector. 1 = Smoker, 0 = Non-smoker

% Plot the data
plot(x(y==1,1), x(y==1,2), 'b+'); hold on;
plot(x(y==0,1), x(y==0,2), 'ro')
legend('Smoker', 'Non-smoker')
xlabel('Age'); ylabel('Blood Pressure')

% Implement the sigmoid function. Complete the code in sigmoid.m. The
% function should perform the sigmoid function of each element in a vector
% or matrix.
g = sigmoid([-10 0.3; 0 10])
% If your implementation is correct you should get the following answer.
% g =
%
%     0.0000    0.5744
%     0.5000    1.0000

% Add intercept term to x and initialize theta
[m, n] = size(x);
X = [ones(size(x,1), 1) x];
initial_theta = zeros(n + 1, 1);

% Now it is time to implement logistic regression in
% costLogisticRegression.m.
[J grad] = costLogisticRegression(initial_theta, X, y);

% You can check if your implementation is correct by comparing the
% gradients with the analytical gradients.
numgrad = checkGradient(@(p) costLogisticRegression(p, X, y), ...
    initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Instead of using our own implementation of gradient descent
% (gradentDescent.m) we will use the pre-built MATLAB function fminunc
% which sets the learning rate alpha automatically.
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(costLogisticRegression(t, X, y)), initial_theta, ...
    options);

% Plot data and decision boundary
plot(X(y==1,2), X(y==1,3), 'b+'); hold on
plot(X(y==0,2), X(y==0,3), 'ro');
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y, 'Color', 'k', 'Linewidth', 2)
xlabel('Age'); ylabel('Blood Pressure')
legend('Smoker', 'Non-smoker')

% Now we use the learned logistic regression model to predict the
% probability that a patient with age 32 and blood pressure 124 is a
% smoker.
prob = sigmoid([1 32 124] * theta) % Should return a value around 0.35
% So the probability that this patient is a smoker (the positive class y=1)
% is 35% and the probability that this patient is a non-smoker is
% 1-prob = 65%. This means that we classify this patient as a non-smoker.
% However, if you look at the plot it looks like this is a
% missclassification because the training data contains many smokers around
% that age and blood pressure. It seems like the assumption of a line is
% not enough to capture enough variations in the data.

% ====================== REPORT ==============================
% Show the final code in sigmoid.m and costLogisticRegression.m. What value
% of diff and prob did you get? Include the plot of the data with the
% logistic regression decision boundry.
% ============================================================

%% ============== Part 7: Polynomial features ==================
% NOTE: THIS PART IS ONLY REQUIRED FOR PASS WITH DISTINCTION. YOU CAN SKIP
% THIS PART IF YOU ONLY AIM FOR PASS.

% In this part we introduce polynomial features in order to fit a curve
% instead of a line to the data. You might need to change
% costLogisticRegression.m so that it works for more than two variables.
degree = 2;
Xpoly = mapFeature(X(:,2), X(:,3), degree);
% Feature normalization becomes important when using polynomic features
[Xpoly(:,2:end) mu sigma] = featureMeanNormalize(Xpoly(:,2:end));
initial_theta = zeros(size(Xpoly, 2), 1);
lambda = 0; % regularization parameter lambda to 1

% Set options and optimize theta
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag, output] = fminunc(@(t)(costLogisticRegression(t, ...
    Xpoly, y, lambda)), initial_theta, options);

% Plot data and Boundary
plot(X(y==1,2), X(y==1,3), 'b+'); hold on
plot(X(y==0,2), X(y==0,3), 'ro');
u = linspace(15, 60, 50);
v = linspace(95, 150, 50);
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        temp = (mapFeature(u(i), v(j), degree) - [1 mu])./[1 sigma];
        z(i,j) = sigmoid(temp*theta);
    end
end
z = z'; % important to transpose z before calling contour
contour(u, v, z, [0.5 0.5], 'Color', 'k', 'LineWidth', 2)

% ====================== YOUR CODE HERE ======================
% Calculate the probability that a patient with age 32 and blood pressure
% 124 is a smoker.
%prob = ...
% ============================================================

% ====================== REPORT ==============================
% FOR PASS WITH DISTINCTION: Vectorize and show the final code in
% costLogisticRegression.m. Include the plot of the data of polynomial
% features with the logistic regression decision boundry. Report what the
% probability that a patient with age 32 and blood pressure 124 is a
% smoker.
% ============================================================
