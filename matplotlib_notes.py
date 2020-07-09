import matplotlib.pyplot as plt
import numpy as np

### Plot Creation & Display ###
# Creating x-axis values
x = np.arange(0, 3 * np.pi, 0.1)
# Creating y-axis values    
y_sin = np.sin(x)                   
y_cos = np.cos(x)
# Plotting values
plt.plot(x, y_sin)                  
plt.plot(x, y_cos)
# Labeling
plt.title('Sine & Cosine')
plt.xlabel('x-axis')                   
plt.ylabel('y-axis')
plt.legend(['Sine', 'Cosine'])
# Displaying
plt.show()             
# Subplots
plt.subplot(2, 1, 1)    # set up subplot grid with height 2 and width 1, set first subplot as active
plt.plot(x, y_sin)
plt.title('Sine')
plt.subplot(2, 1, 2)    # set second subplot as active
plt.plot(x, y_cos)
plt.title('Cosine')
plt.show()