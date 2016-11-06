# created for homework in my Machine Learning course. Some of the algorithms and cost functions were adapted from other data scientists' blogs.

sigmoid = function(z){
  1 / (1 + exp(-z))
}
#cost function
cost = function(X, y, theta){
  m = nrow(X)
  hx = sigmoid(X %*% theta)
  (1/m) * (((-t(y) %*% log(hx)) - t(1-y) %*% log(1 - hx)))
}

cost2 = function(X, y, theta){
-1 * sum((Y)* log(sigmoid(X %*% theta)) + (1-Y)*log(1-sigmoid(X %*% theta)))
}

#gradient
grad = function(X, y, theta){
  m = nrow(X)
  hx = sigmoid(X %*% theta)
  (1/m) * (t(X) %*% (hx - y))
}

# hypothesis 
h = function(X,theta) {
  return( sigmoid(X %*% theta) )
} # h(x,th)

#set up Hessian
Hessian = function (X,y,theta,m) {
  return (1/m * t(X) %*% X * diag(h(X,theta)) * diag(1 - h(X,theta)))
} # H(x,y,th,m)

######################################################################## GRADIENT DESCENT
X <- readgeno("one.geno")
Y <- readpheno("two.pheno")
alpha = 0.00001
maxinteration = 50
#lm(y ~ X)
Y <- as.matrix(Y)
X <- t(as.matrix(X))
X <- cbind(1,X)
theta <- rep(0,ncol(X))
m = nrow(X)

costvector <- c()
for (i in 1: maxinteration){
  theta <- theta - alpha * grad(X, Y, theta)
  costvector<- append(costvector, cost2(X,Y, theta))
  print(i)}

plot(costvector, ylab="NLL", xlab= "Iteration")
title(main="NLL for Log. Regression Gradient Descent (Step= 0.00001)", col.main="red", font.main=4)

########################################################################### HESSIAN
X <- readgeno("one.geno")
Y <- readpheno("two.pheno")
Y <- as.matrix(Y)
X <- t(as.matrix(X))
X <- cbind(1,X)
theta <- rep(0,ncol(X))
m = nrow(X)

# define another cost function for Hessian
cost3 = function (x,y,th,m) {
  return( 1/m * sum(-y * log(h(x,th)) - (1 - y) * log(1 - h(x,th))) )
} # J(x,y,th,m)

# derivative of J (gradient)
grad2 = function (x,y,th,m) {
  return( 1/m * t(x) %*% (h(x,th) - y))
} # grad(x,y,th,m)


alpha = 0.0
maxinteration = 50
costvector2 <- c()


#iterate to 7th iteration (point at which iteration achieves lowest cost)
for (i in 1:7) {
  costvector2 <- append(costvector2, cost3(X,Y,theta,m)) # appends results to a costvector2 object
  theta = theta - solve(Hessian(X,Y,theta, m)) %*% grad2(X, Y,theta, m) 
}


  #plot Newton's method
  plot(costvector2, xlab="Iterations", ylab="Cost")
  title(main="NLL for Log. Regression Newton's Method", col.main="blue", font.main=4)
  
