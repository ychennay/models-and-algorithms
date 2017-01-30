Originally based upon http://nbviewer.jupyter.org/github/tfolkman/learningwithdata/blob/master/Logistic%20Gradient%20Descent.ipynb

Derivation of a logistic regression:

Let's assume we are working with the famous iris dataset. Our two X variables are 1) Sepal Length and 2) Sepal Width.

The equation for a logistic regression starts with a sigmoid function, which can be coded as 1 / (1+math.e(-x)). This will produce a set
of y(x) values that range from 0 to 1. At x = 0, y(x) will equal exactly 0.5.

The input (x) can be defined as a linear combination:

x(sepal_width, sepal_length) = beta_0 + beta_1 * sepal_width + beta_2 * sepal_length

x = input value for sigmoid function
beta_0 = our "intercept" or bias
beta_1 = coefficient for sepal width
beta_2 = coefficient for sepal length

The maximum likelihood probability for setosa class is:

for x in range(len(dataset)):
  newest_h(x) = 1/ (1+ math.e**-(beta_0 + beta_1 * sepal_width(x) + beta_2 * sepal_length(x)))
  h(x) = h(x) * newest_h(x)

Since the only other class in question here is versicolor, the maximum likelihood probability for setosa class is:
for x in range(len(dataset)):
  newest_h(x) = [1- 1/ (1+ math.e**-(beta_0 + beta_1 * sepal_width(x) + beta_2 * sepal_length(x)))]
  h(x) = h(x) * newest_h(x)

Written more simply, it is probability(versicolor) = 1-h(x).

Our expression that we are trying to maximize, therefore is h(x) * [1- h(x)]. This represents the combination of parameters (beta_0,
beta_1, beta_2) that maximize the likelihood of each sample being setosa or versicolor. Moreover, maximizing the positive expression is
the same as minimizing the negative function, and for the sake of our gradient descent algorithm, we will be using a minimization function
to find out when our derivative approaches 0.

Before we move on to the next section of the walkthrough, an important concept to understand is the behavior of products and logs. 

In python, math.log(x,y) = z means that y**z = x, or in common English, that y raised to the zth power will equal x. Y is the base, and z is
the exponent in this case. For our logistic regression, it is important to understand that a product can be converted into sums by taking the log
of a product x*y:

product = x * y
math.log(product, a) = math.log(x, a) + math.log(y, a)

This is a useful way of separating out terms in our gradient descent function. We can apply this same reasoning to our h(x) equation:

h(x) = 1/ (1+ math.e**-(beta_0 + beta_1 * sepal_width(x) + beta_2 * sepal_length(x)))

product = h(x) <-- setosa max likelihood * 1- h(x) versicolor max likelihood

math.log(product, natural_e) = math.log(h(x), natural_e) + math.log(1-h(x), natural_e)
new tentative maximum likelihood function = math.log(h(x), natural_e) + math.log(1-h(x), natural_e)

But how will our function make sure that it is adjusting to the actual real life values (how do we use the actual class labels to "train"
the logistic regression?). We include the y values in our function. Let y = 1 be setosa and y = 0 be versicolor. We add these to our sums so that 
our function adjusts for each class example to consider how close it was to the positive sample:

max_prob = y_value *(math.log(h(x), natural_e)) + (1-y_value) *( math.log(1-h(x), natural_e))

What happens if the first observation is a setosa? Then y =1, and the only part that is considered is the first term:

max_prob = y_value *(math.log(h(x), natural_e))
max_prob = math.log(h(x), natural_e) <-- since y-value = 1

In the equation above, we are looking to maximize the probability. How can we do this? Well, the higher the value of h(x), the larger math.log(h(x), natural_e)
will be as well. The greater that expresion is, the larger the max_probability will be!

Remember that we are trying to find the minimum of the negative. Where does a minimum (at least a local minimum) occur in a function?
When the derivative of that function approaches 0. Taking the derivative of a log function is as follows:

f(x) = math.log(x, b)
f'(x) = 1 / (x * ln(b))

In this example, if b = natural_e (~2.718), then ln(natural_e) = 1. Therefore, the derivative of math.log(x, natural_e) is 1/x. As a result, for
each of the observations in the data set, we get the following expression:

max_prob' = y_i / h(x_i) + (1 - y_i) / (1 - h(x_i))

The Quotient Rule for Derivatives states that

h(x) = f(x) / g(x)
h'(x) = [g(x) * f'(x) - f(x) * g'(x)] / (g(x) ** 2)

Remember the original sigmoid function?Our h(x) = 1 / (1 + e**-x). If we apply the quotient rule, we will get the following
 derivative for h'(x):
 
h'(x) = (e**-x) / (1 + e**-x)**2 
h'(x) = [1 / (1 + e**-x)] * [1 - 1 /(1 + e**-x)]
h'(x) = h(x) * (1 - h(x))

In plain English, this is saying that the rate of change of our hypothesis will equal the hypothesis for a given observation multiplied by the 1 - hypothesis inverse.

Now, let's look at the other part of the input:

x = beta_0 + beta_1 * sepal_width + beta_2 * sepal_length

Let's take the derivative of x with respect to beta_0. Assume that everything else to the right of the righthanded expression is a constant.
Therefore, when we take the derivative, x' = 1.

Now let's find the overall derivative with respect to beta_0 for setosa:

y_i * log(h(x_i), natural_e) # original expression
y_i * log(h(x_i), natural_e) * h(x_i) * (1 - h(x_i)) # derivative of h(x)
y_i * h(x_i) * (1 - h(x_i)) / (h(x_i) # derivative of log(x) 

The same is true for versicolor, with the exception of replacing y_value with (1 - y_value).

When this is simplified, it becomes y_i - y_i * h(x_i) - h(x_i) + y_i * h(x_i).

This can be ultimately simplified even further to (y_i - h(x_i)) * 1 for i = range(len(dataset)).

The partial derivatives for the beta_1 and beta_2 are similar. However, instead of 1 as the derivative with respect to beta_0, the new derivative for x with respect to beta_1
is sepal_width, and with respect to beta_2 is sepal_length.

Then the three relevant partial derivatives (gradient) are:
h(x_i) - y_i for beta_0
(h(x_i) - y_i) * sepal_width_i for beta_1 (the coefficient of sepal width)
(h(x_i) - y_i) * sepal_length_i for beta_2 (the coefficient of sepal length)

The way to update our betas (coefficients) using gradient descent is through the following equation:
beta_i = beta_i - (alpha * gradient)

The alpha is the learning rate. Too high a learning rate and the algorithm overshoots, utlimately the error terms to infinity. Too small
an alpha, and the algorithm converges too slowly. 

Why are we subtracting? Because think about what we are looking to get. We gradient represents the direction of increase, and since we want to find the direction
of decrease, we go opposite that direction. We are checking which direction to move the beta value in order to get to the minimum (we move towards the direction
that minimizes our beta values). 
