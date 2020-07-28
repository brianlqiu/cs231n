import numpy as np
import math
"""
### Backpropagation ###
- [Backpropagation] - a way of computing gradients of expressions through recursive chain rule
- Given some function f(x) where x is a vector of inputs, we are interested in computing the gradient of f at x

## Simple Expressions & Interpretation of Gradient ##
- Derivative of an expression on a variable tells you the sensitivity of the whole expression its value
    - i.e. if f(x,y) = xy, f'x(x,y) = y, f'y(x,y), let x = 4, y = -3
           if we change x by some small amount, we decrease the whole function by 3 times the value f'x(4,-3) = -3
           if we change y by some small amount, we increase the whole function by 4 times the value f'y(4,-3) = 4
- Gradient is the vector of partial derivatives
    - grad(f) = [f'x(x,y), f'y(x,y)]
- Examples of grad
    - f(x,y) = x + y -> grad(f) = [1, 1] 
        - Increasing x or y increases f, rate of increase independent of values of x or y
    - f(x,y) = max(x,y) -> grad(f) = [1(x >= y), 1(y >= x)]
        - nax(x,y) would return x or y, so derivative would return indicator function
        - Since derivatives are infinitesimally small changes, we don't have to worry about the other variable getting 
          larger

## Compound Expressions with Chain Rule ##
- Use backpropagation to compute chain rule derivative
"""
# Let f(x,y,z) = (x + y)z
x = -2
y = 5
z = -4

# Let q = x + y
q = x + y
f = q * z

# Perform backpropagation in reverse order
# First backprop through f = q * z
dfdz = q            # df/dz = q
dfdq = z            # df/dq = z
# Then backprop through q = x + y
dfdx = 1.0 * dfdq   # df/dx = dq/dx * df/dq = 1.0 * df/dq
dfdy = 1.0 * dfdq   # df/dy = dq/dy * df/dq = 1.0 * df/dq

"""
## Intuitive Understanding ##
- Imagine the above example as a circuit, with inputs x, y, z and gates being the operations
- Each gate can calculate the output value (i.e. x + y) and it's local gradient with respect to inputs (i.e. 1)
- Walking through the example
    - Add gate received input (-2, 5) and computed output 3; since gate is addition, local gradient is 1 (1, 1)
    - Rest of circuit computes final value, -12
    - During backward pass in which chain rule is applied recursively, add gate learns that gradient was -4
        - Circuit wants the output of add gate to be lower with a force of 4
        - Add gate takes the gradient and multiplies it to all local gradients for its output
- My understanding:
    - Inputs are taken through the circuit and a result is calculated
    - With calculus, we can calculate the gradient of each gate
        - The multiply gate has 2 variables, q and z, and the gradients are dq = z (-4) and dz = q (3) since it's multiplication
    - Thus, we can conclude the gradient for z = -4, since there are no more nodes after z
    - Now tackling the add gate
        - We know that the add gate has 2 variables, x and y, and the gradients are dx = 1 and dy = 1 for addition
    - We have to spread that gradient that we calculated for q (-4) across the gradients, so dx = 1 * -4 and dy = 1 * -4

## Modularity: Sigmoid Example
- Any differentiable function can act as a gate and can be decomposed/grouped into multiple/single gates
- Ex: f(w,x) = 1 / (1 + e**(-(w0x0 + w1x1 + w2))) - a 2D neuron using the sigmoid activation function
- Composed of the following gates:
    - f(x) = 1 / x  -> f'(x) = -1 / x**2
    - f(x) = c + x  -> f'(x) = 1
    - f(x) = e ** x -> f'(x) = e ** x
    - f(x) = ax     -> f'(x) = a   
- Sigmoid function: sigma(x) = 1 / (1 + e ** -x)
- Derivative = (e ** -x) / ((1 + e ** -x) ** 2) -> (1 - sigma(x))sigma(x)
"""
# Backpropogation for Sigmoid
w = [2, -3, -3]
x = [-1, -2]

# Forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))          

# Backward pass
ddot = (1 - f) * f                      # gradient on dot variable, using sigmoid gradient derivation
dx = np.array([w[0], w[1]]) * ddot      # backprop into x
dw = np.array([x[0], x[1], 1]) * ddot   # backprop into y

"""
## Staged Backprop ##
"""
# Ex: f(x,y) = (x + sigma(y)) / (sigma(x) + (x + y)**2), x = 3, y = -4
x = 3
y = -4

# Forward pass
sigy = 1.0 / (1 + math.exp(-y))
num = x + sigy                  # numerator
sigx = 1.0 / (1 + math.exp(-x))
xpy = x + y
xpysqr = xpy**2
den = sigx + xpysqr
invden = 1 / den                # inverse denominator
f = num * invden

# Backwards pass
# For every variable, we have same variable that holds the gradient with respect tothat variable
# backprop f = num * invden
dnum = invden
dinvden = num 
# backprop invden = 1 / den
dden = -1.0 / den**2 
# backprop den = sigx + xpysqr
dsigx = 1.0 * dden
dxpysqr = 1.0 * dden
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr
# backprop xpy = x + y
dx = 1.0 * dxpy
dy = 1.0 * dxpysqr
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += (1 - sigx) * sigx * dsigx
# backprop num = x + sigy
dx += 1.0 * dnum
dsigy = 1.0 * dnum
# backprop sigy + 1.0 / (1 + math.exp(-y))
dy += (1 - sigy) * sigy * dsigy

"""
- Good to cache variables to avoid recomputing in backwards pass
- Gradients add up at forks (look at +=)

## Patterns in Backwards Flow ##
- Add gate always takes gradient and distributes it equally to all inputs
- Max gate routes the gradient to exactly one of its input (input with higher value during forward pass)
- Multiply gate are local gradients switched and multiplied by gradient

## Gradients for Vectorized Operations ##
"""
# Matrix-matrix multiply gradient
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)

# Forward Pass
D = W @ X

# Backwards Pass
dD = np.random.randn(*D.shape)  # suppose we had gradient of D from above, should be (5, 3)
dW = dD.dot(X.T)                # we need to transpose for dimensional analysis, dot product-ing (5, 3) and (3, 10), to original size of W (5, 10)
dX = W.T.dot(dD)                # same idea, but we need size (10, 3), so transpose W 

