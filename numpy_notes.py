import numpy as np

### Array Creation ###
a = np.array([[1,2], [3,4], [5,6]]) # normal array creation
print(a.shape)                      # (3,2)
a = np.zeros((2,2))                 # create 2x2 matrix of 0s           [[0 0] [0 0]]
a = np.ones((1,2))                  # create 1x2 matrix of 1s           [1 1]
a = np.full((2,2), 7)               # create 2x2 matrix of 7s           [[7 7] [7 7]]
a = np.eye(2)                       # create 2x2 identity matrix        [[1 0] [0 1]]
a = np.random.random((2,2))         # create 2x2 matrix of random vals  

### Array Indexing ###
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
## Slicing ##
# Note: Slicing is a reference to an array, does not return a new array
print(a[:2,1:3])                                # slice rows 0-1 and cols 1-2       [[2 3] [6 7]]
print(a[1,:])                                   # slice out row 1 in rank 1         [5 6 7 8]
print(a[[1],:])                                 # slice out row 1 in rank 2         [[5 6 7 8]]
## Integer Array Indexing ##
print(a[[0,1,2], [0,1,0]])                      # slice out a[0,0], a[1,1], a[2,0]  [1 5 3]
print(a[np.arange(3), np.array([0,2,1])])       # selecting elements from each row  [1 6 10]
print(a[np.arange(3), np.array([0,2,1])] + 10)  # mutating elements from each row   [11 16 20]  
## Boolean Array Indexing ## 
# Create a new array containing only elements that fulfill some condition 
print(a[a > 6])                                 # slice out only elements > 6       [7 8 9 10 11 12]

### Datatypes ###
# Numpy auto chooses datatype, but you can specify #
a = np.array([1,2], dtype=np.int64) # force type w/ dtype

### Operations ###
## Element Wise Operations ##
a = np.array([[1,2], [3,4]], dtype=np.float64)
b = np.array([[5,6], [7,8]], dtype=np.float64)
print(a + b)                    # element wise addition         [[6 8] [10 12]]
print(a - b)                    # element wise subtraction      [[-4 -4] [-4 -4]]
print(a * b)                    # element wise multiplication   [[5 12] [21 32]]
print(a / b)                    # element wise division         [[0.2 0.333] [0.429 0.5]]
print(np.sqrt(a))               # element square root           [[1 1.414] [1.732 2]]
## Dot Product ##
print(a @ b)                    # dot product                   [[19 22] [43 50]]
## Sum ##
print(np.sum(a))                # sum of all elements           10
print(np.sum(a, axis=0))        # sum of each column            [4 6]
print(np.sum(a, axis=1))        # sum of each row               [3 7]
## Transpose ##
# Commonly used to transpose row vectors into column vectors
print(a.T)                      # transpose                     [[1 3] [2 4]]
## Broadcasting ##
"""
BROADCASTING RULES
- If arrays are not the same rank, prepend shape of lower rank array with 1s until both shapes have same length
- 2 arrays are compatible in a dimension if they have same size in dimension or if one array has size 1 in dimension
- Arrays can be broadcasted if they are compatible
- After broadcast, arrays behave as if it had shape equal to the elementwise max dimensions of the 2 input arrays
- In any dimension where one array has size 1 and the other array has size greater than one, the first array behaves   
  like it was copied along that dimension
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
b = np.array([1,0,1])
c = np.empty_like(a)            # create an empty matrix w/ same dimensions of a
c = a + b                       # mutating array by cols/rows   [[2 2 4] [5 5 7] [8 8 10] [11 11 13]]
# Reshaping #
a = np.array([1,2,3])
b = np.array([4,5])
print(np.reshape(a, (3,1)) * b) # reshapes a to a column vector, broadcast against b    [[4 5] [8 10] [12 15]]
## Stacking ##
a = np.tile(b, (4,1))       # stacking 4 copies of b on top of each other





 


