function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%theta(1) should not be regularized

for j=1:m
    J=J+(-y(j)*log(sigmoid(X(j,:)*theta))-(1-y(j))*log(1-sigmoid(X(j,:)*theta)));
end

J=J/m;
theta_sum=0;
for k=2:size(theta,1)
    theta_sum=theta_sum+theta(k)^2;
end
J=J+lambda/(2*m)*theta_sum;


for i=1:size(theta,1)
    grad_sum=0;
    for j=1:m
        grad_sum=grad_sum+(sigmoid(X(j,:)*theta)-y(j))*X(j,i);
    end
    grad_sum=grad_sum/m;
    if i==1
        grad(i)=grad_sum;
    else
        grad(i)=grad_sum+lambda/m*theta(i);
    end
end   
    
% =============================================================

end
