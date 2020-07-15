import numpy as py
"""
### Linear Classification ###
- [Score function]  - function that maps raw data to class scores
- [Loss function]   - quantifies the agreement between predicitons and truth

## Parametrized Mapping from Images to Label Scores ##
- Score function that maps pixel values of an image to confidence scores for each class
- Assume we have <N> examples of dimensionality <D> with <K> categories/labels
- The score function is a function f: R**D -> R**K (function that maps the raw image
  pixels to class scores)
# Linear Classifier #
- A linear mapping:
    f(xi,W,b) = W(xi) + b
    - xi    = the flattened image   shape=[D, 1]
    - W     = weights               shape=[K, D]
    - b     = bias vector           shape=[K, 1]
- Matrix mulitplication <W(xi)> evaluates 10 separate classifiers in parallel, where 
  each row is a different classifier for each category
- Goal is to optimize the parameters to match the truth labels as closely as possible
- We can discard the training set afterwards and just keep the parameters
- Classifying test image is single matrix multiplication and addition, much faster than comparing all
# Interpreting a Linear Classifier #
- A linear classifier computes the score of a class as a weighted sum of all of its pixel
  values across all 3 color channels
- Function prefers certain colors at certain positions in the image (more blue on the
  outer pixels could indicate a ship)
# Images as high-dimensional points
- Imagine each image is a data point, the classifier essentially draws a line to 
  classify images (where the weights are the coefficients and the bias vector is the y intercept)
# Linear classifiers as template matching
- Each row of <W> corresponds to a template/prototype
- The score of each class of an image is obtained by taking the dot product one by one to find the one that fits best
- We are creating templates as the weights, which are like an average of all of the images in the training set
    - In kNN, we use the whole training set to calculate distance; think of this using the template/prototype instead
      of the whole thing
# Bias Trick
- Trick to represent the two parameters <W, b> as one
- Just add b to W as another column, functionally identical when taking the dot prodcut
# Image data preprocessing
- Center the data by computing the mean pixel data and subtracting from each feature (can perform additional with 
  dividing by std dev)

## Loss Function ##
# Multiclass Support Vector Machine Loss #
- [Multiclass Support Vector Machine (SVM) loss]
- The SVM loss wants the correct class for each image to have a score higher than the incorrect classes by some fixed 
  margin <delta> (want confident predictions)
- Assume we are given vectorized image <x[i]> and the label <y[i]> that specifies the index of the correct class
    - Score function takes these variables and computes the vector f(x[i], W) of class scores, abbreviated as <s>
    - The score for the jth class is the jth element: s[j] = f(x[i], W)[j]
    - Multiclass SVM for the ith example is formalized as:
"""
# i         = example index
# x         = matrix containing the images
# w         = matrix of weights
# delta     = some predefined margin integer
# yind      = int representing the index of the correct class (0 = first class, 1 = second class, ...) 
def loss(i, w, x, delta, yind):
    # if no loss (returns 0), that means the score of the correct class exceeds 
    # delta for all other incorrect classes
    # otherwise, loss is comprised of the sum of how close other classes get to the 
    # correct score                              

    # 0 gets broadcasted
    # np.delete(w, yind).T @ x[i] gives you a vector of scores for each class EXCLUDING the correct one
    # w[yind].T @ x[i] gives you the score for the correct class and gets broadcasted to subtract against all
    # the incorrect ones
    # add delta to see if the correct class has a score that exceeds the incorrect one by some margin
    return np.sum(np.maximum(0, (np.delete(w, yind).T @ x[i]) - (w[yind].T @ x[i]) + delta)) 
"""
- [Hinge loss] - the threshold at 0 (max(0, ...)) is called the hinge loss
    - [Squared hinge loss] - max(0, ...)^2, penalizes violated margins more heavily
# Regularization
- W isn't necessarily unique, since both the correct and the incorrect ones get multiplied by the same W, we can 
  multiply W by any scalar and achieve the same loss (i.e. if we multiplied W by 2, we would get same loss)
- [Regularization penalty] - encode some preference for a certain set of weights over others
- [L2 norm] - regularizaton penalty that discourages large weights through elementwise quadratic penalty over all   
  parameters
"""
# W = weights
def reg_penalty(W):
    return np.sum(np.square(W)) # square each element, add all elements up
"""
- Full multiclass SVM loss becomes: 
L = (1 / N) * Sum(i, num_examples, DL) + lambda * lambda * RW
or
"""
# N         = number of training examples
# x         = matrix of flattened images
# W         = weights
# delta     = predetermined margin
# y         = array of ints representing the correct class for the index (i.e if y[2] = 0, then the second image is 
#             of class 0)
# lambda    = some mutliplier for regularization loss
def full_loss(N, x, W, delta, y, lambd):
    data_loss = 0                                       
    for i in range(N):                                  # sum up total loss for ALL images
        data_loss += loss(i, W, x, delta, y[i])
    return (1 / N) * data_loss + lambd * reg_penalty(W)
"""
- Hyperparameter lambda usually determined through cross-validation    
- Penalizing large weights tends to improve generalization
    
## Practical Considerations ##
# Setting Delta
- Delta can be safely set to 1.0 in all cases
- Delta doesn't actually matter, because W can be shrunk or expanded upon
- Rely on regularization strength lambda to control size of weights

## Softmax Classifier ##
- Softmax returns probabilities instead of scores
- [Cross-entropy loss] replaces hinge loss
    L[i] = -f[y[i]] + log(Sum(j, e**f[j])) 
    f[j] = jth element of the vector of class scores <f>
- [Softmax function] - takes a vector of arbitrary real-valued scores in <z> and squashes it to a vector of values 
  between 0 and 1 that sum to 1
    f(j, z) = (e ** z[j])/(Sum(k, e ** z[k]))
- [Cross-entropy] - the cross entropy between a true distribution p and an estimated distribution q is defined as:
    H(p,q) = -Sum(x, p(x)log(q(x)))
    - Minimzes the cross-entropy between estimated class probabilities (q) and the true distribution (where p = some 
      sequence of 0's and 1's, actual class probabilities)
- P(y[i] | x[i];W) = (e ** f[y[i]])/(Sum(j, e**f[j])) - normalized probability assigned to the correct label y[i] given 
  the image x[i] and parameterized by W
    - Exponentiating gives unnormalized probabilities (since the vector f is log values) and dividing normalizes
    - [Maximum likelihood estimation (MLE)] - minimizing the negative log likelihood of the correct class
- Practical issues: numeric stability
    - e ** f[y[i]] and Sum(j, e**f[j]) can be very large, numerically unstable
    - Instead, use an exponent trick to rewrite function as:
        e**(f[y[i]] + log(C)) / Sum(j, e**(f[j] + log(C)))
        where C can be any constant
    - Common choice is log(C) = -max(f[j]) (make it so that the highest number is 0)

## SVM vs Softmax ##
- Softmax provides probabilities
    - Probability distribution depends on the regularization strength <lambda>
        - The higher lambda is, the tighter the output probabilities get (almost to uniform!)
- SVM and Softmax are comparable in performance
    - SVM is a more local objective
        - Doesn't care about the distribution of scores as long as one surpasses the margin, computes loss ignoring this
            - Ex: can have [10, -100, -100] or [10, 9, 9] and both losses would be 0 (if the margin delta = 1)
    - Softmax never satisfied
        - Cares more about the distribution of scores, will always provide a loss
            - Ex: with previous ex, the [10, 9, 9] would have much higher loss
"""