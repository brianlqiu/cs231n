"""
### Optimization ###
- [Optimization] - the process of finding the set of parameters <W> that minimize the loss function

## Visualizing the Loss Function ##
- Loss functions are usually defined over very high-dimensional spaces
    - Ex: in CIFAR-10, the weight matrix is (10, 3073) (10 categories, 3073 pixel values)
- Gain intuition by slicing along the high-dimensional space along rays or allong planes
    - Generate a random weight matrix/direction <W[1]> and compute the loss along this direction by evaluating 
      L(W + aW[1]) for differnt values of a
        - Generates a 2D plot with <a> as the x-axis and the loss function as y-axis
    - You can do the same thing with 2 variables (L(W + aW[1] + bW[2])) to get a 3D slice
    - Example: Consider a dataset with 3 1-dimensional points and 3 classes. The full SVM loss (w/o regularization) is:
        L[0] = max(0, w[1]x[0] - w[0]x[0] + 1) + max(0, w[2]x[0] - w[0]x[0] + 1) // loss for each data point
        L[1] = max(0, w[0]x[1] - w[1]x[1] + 1) + max(0, w[2]x[1] - w[1]x[1] + 1)
        L[2] = max(0, w[0]x[2] - w[2]x[2] + 1) + max(0, w[1]x[2] - w[2]x[2] + 1)
        L = (L[0] + L[1] + L[2]) / 3                                             // average loss over all data points
        Each L[0], L[1], L[2] create their own 2D plot, with loss as y-axis and the weights as x-axis
        The total loss is an "average" of the plot of the others, usually convex

## Optimization ##
- Loss function lets us quantify the quality of a set of weights W
- Optimization is to find W that minimizes the loss function

# Strategy 1: Random Search #
- Bad idea
- Generate random weights and track the one that has lest loss
"""
bestloss = float("inf")                     # assign highest possible value
for num in range(1000):                     # generate 1000
    W = np.random.randn(10, 3073) * 0.0001  # generate random parameters
    loss = L(X_train, Y_train, W)           # L = loss function, X_train = (3073, 50000) dataset where each column is 
                                            # an example, Y_train = (1, 500000) vector containing correct labels for 
                                            # each example
    if loss < bestloss:                     # keep track of lowest loss
        bestloss = loss
        bestW = W
"""
- We want iterative refinement - start with a random W and iteratively refine

# Strategy 2: Random Local Search #
- Start out with a random W, generate random changes to W (deltaW) and if the loss at W + deltaW is lower, perform 
  update
"""
W = np.random.randn(10, 3073) * 0.001                   # random start
bestloss = float("inf")
for i in range(1000):
    step_size = 0.0001                                  # determines how big the change to W is 
    Wtry = W + np.random.randn(10, 3073) * step_size    # make the attempt
    loss = L(Xtr_cols, Ytr, Wtry)                       # calculate loss on the attempt
    if loss < bestloss:                                 # store the new weight <Wtry> if it has lower loss
        W = Wtry
        bestloss = loss
"""

# Strategy 3: Following Gradient #
- The downside of strat 2 is the random part
- We don't have to rely on randomness, we can calculate the best direction (gradient)
- Gradient is te vector of partial derivatives in each dimension
# Computing Gradient #
# Numerical gradient
"""
# f = function
# x = vector to evaluate gradient on (the multidimensional point)
# returns the gradient of f at x
def numerical_gradient(f, x):   
    fx = f(x)                                                           # evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.0001

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])    # iterate through all indices in x
    while not it.finished:
        ix = it.multi_index                                             # get current index
        old_value = x[ix]                                               # store in old_value
        x[ix] = old_value + h                                           # increment the current index by h
        fxh = f(x)                                                      # store the new function value at fxh
        x[ix] = old_value                                               # restore old value
        grad[ix] = (fxh - fx) / h                                       # store the gradient in that index direction
        it.iternext()                                                   # go to next
    return grad 
"""
- Iterates over all dimensions, makes a small change <h> and calculates the partial derivative of the loss function 
  along that dimension by seeing how much the function changed
- Note that the definition of derivative is f'(x) = lim_(h->0)(f(x+h) - f(x) / h) but in our function we use h = 0.0001
    - Practically, it works
- In practice, the [centered difference formula] - (f(x + h) - f(x - h)) / 2h works better
"""
def CIFAR10_loss(W):                        # our gradient function only takes functions w/ 1 argumnet
    return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001
df = numerical_gradient(CIFAR10_loss, W)    # get gradient

loss_original = CIFAR_loss(W)               # get original loss
for step_size_log in range(-10, 0):         # try different step sizes
    step_size = 10 ** step_size_log
    W_new = W - step_size * df              # take a step in the gradient (subtract because we want loss to decrease)
    loss_new = CIFAR10_loss(W_new)          # store new loss
"""
- [Step size / Learning rate] - how large of a descent we should take after the gradient
    - Extremely important hyperparameter
- Small step sizes = worse performance (more iterations)
- Large step sizes = more inaccuracy
- Inefficient for large numbers of parameters: we need to perform loss function for each parameter
# Analytic gradient
- Numerical gradient is an approximation, based on the size of h
- Analytic uses calculus to give exact value, and is faster
- However, more error-prone to implement
    - Common to use a [gradient check] - compute analytic gradient and compare to numerical gradient
- SVM loss function for single data point:
    L[i] = Sum(j != yi, max(0, w[j].T * x[i] - w[yi].T * x[i] + delta))
- Differentiate with respect to the weights:
    dw[yi]L[i] = -Sum(j != yi, 1(w[j].T * x[i] - w[yi].T * x[i] + delta > 0)) * x[i] 
    - Gradient with respect to the row of W that corresponds to the correct class
    - 1() function returns 1 if the condition inside is true (if the score for wrong class exceeds the score of the 
      correct class by delta) and 0 otherwise
    dw[j]L[i] = 1(w[j].T * x[i] - w[yi].T * x[i] + delta > 0) * x[i]
    - Gradient with respect to the other rows

## Gradient Descent ##
- Pseudocode:
    while True:
        weights_grad = evaluate_gradient(loss_function, data, weights)
        weights += - step_size * weights_grad
- Core of all neural network libraries

# Mini-batch Gradient Descent #
- Computationally expensive to compute full loss function over entire training set for one parameter update
- Instead, compute gradient over batches of training data

while True:
    data_batch = sample_training_data(data, 256) # get 256 examples
    weights_grad = evaluate_gradient(loss_function, data, weights)
    weights += - sstep_size * weights_grad

- Works well because examples in training data are correlated
    -Example: Consider that 1.2 million images in dataset are just duplicates
        - If you compute gradient descent over all, you would get the same loss as a small subset

- [Stochastic Gradient Descent] - mini-batch gradient but only 1 example
    - Rare because it's computationally more efficient to evaluate for a larger sample rather than a single over again
    - SGD is often the same as minibatch gradient descent colloquially
- Size of batches are a hyperparameter but aren't cross-validated usually
    - Base it on memory constraints or set to some power of 2 (usually vectorized operations work faster on powers of 2)
    
"""