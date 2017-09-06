import numpy as np
from code_linear_classifiers import *

D = load_data()
X1 = D['dat1']
Y1 = D['target1']


# The points from previous exercise are in [0,1]x[0,1]. 
# Rescale them to R=[-1,1]x[-1,1]
X1r = X1.copy()
X1r[:,1] = X1r[:,1]*2 -1
X1r[:,2] = X1r[:,2]*2 -1
plt.axis([-1, 1, -1, 1])

# Plot points
plt.scatter(X1r[:,1],X1r[:,2],c=Y1,cmap=plt.cm.Paired,s=80)

# Plot hyperplane
plt.plot([-1,1],[1,-1],'-r',linewidth=2)
plt.show()
