function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%iterations = 1500;
%alpha = 0.01;
%theta -> 2*1

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    tmp=[0 0];
    for j=1:m
        tmp(1)=tmp(1)+(X(j,:)*theta-y(j))*X(j,1);
        tmp(2)=tmp(2)+(X(j,:)*theta-y(j))*X(j,2);
    end
    theta(1)=theta(1)-alpha/m*tmp(1);
    theta(2)=theta(2)-alpha/m*tmp(2);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
