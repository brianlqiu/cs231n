import numpy as np
"""
### Image Classification ###
- [Image classification] - the task of assigning an input image one label from a fixed set of categories
    - Computers are passed in 3D array of numbers (X, Y, RGB values)

# Challenges #
- [Viewpoint variation]     - an object can be oriented many ways w/ respect to camera
- [Scale variation]         - variations in size
- [Deformation]             - objects can be deformed
- [Occlusion]               - objects can be occluded/obstructed
- [Illumination]            - effects of illumination amplified on pixel level
- [Background clutter]      - objects can blend into their environment

# Data Driven Approach #
- [Data driven approach] - provide the computer with a training dataset of labeled images

# Image Classification Pipeline #
- [Input]       - Input consists of N images, each labeled with one of K different classes (the training set)
- [Learning]    - Training a classifier/creating a model using the training set
- [Evaluation]  - Evaluate the quality of the classifier by asking it to predict labels for a new set of images

"""

"""
## Nearest Neighbor Classifier ##
- CIFAR-10 dataset  - set of 60,000 32x32 pixel images classified as one of
    [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
- Nearest neighbor calculates the pixel differences between images and returns the smallest difference
- Methods of Calculating Pixel Differences
    - L1 distance = np.sum(np.abs(X - Y))
    - L2 distance = np.sqrt(np.sum(np.square(X - Y)))
    - L1 vs L2 - L2 prefers many medium disagreements to one big one
- Poor accuracy (around 38.6%), but better than random

# Classifier Pseudocode #
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10')   # load the data
Xtr_rows = Xtr.reshape(Xtr, 32 * 32 * 3)            # flattens training images into 1D
Xte_rows = Xte.reshape(Xte, 32 * 32 * 3)            # flattens test images into 1D

nn = NearestNeighbor()                              # create classifier
nn.train(Xtr_rows, Ytr)                             # trains classifier on training images & labels
Yte_predict = nn.predict(Xte_rows)                  # test classifier on test images
print(f'Accuracy: {np.mean(Yte_predict == Yte)}')   # print results

"""

# Implementation of Nearest Neighbor Classifier #
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X    # nearest neighbor doesn't actually train, rows are examples, columns are pixels, RGB vals, etc.
        self.Ytr = y    # just store the training data & labels

    def predict(self, X):
        num_test = X.shape[0]                               # get number of images to classify
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)  # empty matrix w/ same datatype as input

        # for each input image, find the nearest training image
        for i in range(num_test):                                 
            # broadcast the input image to subtract against training images, creates 1xnum_test array of L1 distances  
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1) 
            min_index = np.argmin(distances)    # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]      # get the label of the nearest neighbor as the prediction

        return Ypred

"""
## k-Nearest Neighbor Classifier ##
- Instead of using only the nearest image, choose the k nearest images and let them vote
- Higher values of k = less affected by outliers
# Hyperparameter Tuning #
- [Hyperparameters] - variables you can change that affects the performance of the classifier
    - Choice of k, distance function, etc.
- Do not use the test set for tweaking hyperparameters!
    - Risk of overfitting
- Split training set into a smaller training set and a validation set
    - Validation set should be much smaller relative to the training set (i.e. 2% for CIFAR-10)
"""

Xval_rows = Xtr_rows[:1000, :]  # reserve first 1000 for validation
YVal = Ytr[:1000]   
Xtr_rows = Xtr_rows[1000:, :]
Ytr = Ytr[1000:]

validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]: # train same validation set for each value of k, record accuracy
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)         
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print(f'Accuracy: {acc}')
    validation_accuracies.append((k, acc))

"""
# Cross-validation #
- [Cross-validation] - split training set into some number of folds, iterate through each fold treating one as the 
                       validation set and the others as training sets, then average the performance across the folds
- Cross-validation is expensive
- Tend to use 50-90% of training data for training and rest for validation
- The more hyperparameters you have, the larger the validation splits you should consider
- 3, 5, or 10 fold cross-validation is common

# Advantages & Disadvantages of Nearest Neighbor #
- Advantages
    - Simplicity
    - No need to train
- Disadvantages
    - Trade low training time for high computational cost at test time
    - The larger the image size is, the worse it performs
    - Can overly focus on color or background rather than identity, since these make up most of the image

# Applying kNN in Practice #
1. Preprocess data through data normalization (have 0 mean and unit variance)
2. If data is high-dimensional, try using a diensionality reduction technique (PCA, NCA, random projections)
3. Split your training data randomly into train/val splits
    - Rule of thumb 70-90% of data should go to training split
    - Depends on number of hyperparameters (more = more validation = less training)
    - IF size of validation data is small, try to use cross-validation
4. Train and evaluate hte kNN classifier on validation data/folds for amny choices of k and across difference 
   distance functions
5. If kNN taking too long, consider using an Approximate Nearest Neighbor library (like FLANN) to accelerate retrieval
   at cost of accuracy
6. Take note of hyperparameters that give the best results, best not to use validation data in final classifier

"""
