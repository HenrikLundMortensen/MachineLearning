# ======================================== QUESTION 1 ========================================

# NumPy is a Python library for efficient computations. 
# For this question we use it to:
#     1. represent the vectors 'theta' and 'x' as arrays.
#     2. compute dot product and sum of 'theta' and 'x'. 
import numpy as np

# Represent the vectors as arrays. 
x =  np.array([1,2,3]) 
theta = np.array([2,3,4])

# Print the NumPy array representation of 'x' and 'theta' and their shapes. 
print('theta:\n', theta)
print('x:\n', x)
print('shapes:', x.shape, theta.shape) 
print('\nDo calculation by hand and check with np.dot or np.sum and *\n')

# Make a variable called 'dot_prod' and assign it the dot product of 'x' and 'theta'. 
dot_prod = np.dot(theta,x)

print('Dot prod is {0}'.format(dot_prod))



# ======================================== QUESTION 2 ========================================

print('Compute the outer product first by hand, then check with np.outer.')
print('You can use np.dot as well if you make \'x\' and \'theta\' into matrices of shape 3,1 with np.expand_dims\n')

# Make a variable called 'outer_prod' and assign it the outer product of 'x' and 'theta'.
outer_prod = np.outer(x,theta)
### YOUR CODE HERE
### END CODE

# Prints the outer product and its shape. 
print('Outer Product Shape Is', outer_prod.shape)
print('Outer Prodcut Is:\n', outer_prod)

print('\nUsing np.expand_dims:')
print(np.expand_dims(x, axis=1) @ np.expand_dims(theta, axis=1).T) #The @ symbol is the matrix multiplication operator

# ======================================== QUESTION 3 ========================================2

# Ensure plots are in-line
# %matplotlib inline

# MatplotLib is a library for plotting data. 
# Here we use it to:
#     1. plot the hyperplane 'theta.T*x=0'
#     2. plot the plus indicator at either (1,1) or (-1, -1). 
import matplotlib.pyplot as plt

# Set text font to make the plot look nice. 
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# The line segment is represented by 'left_point' and 'right_point'. 
left_point = [-2, None]
right_point = [2, None]
plus_pos = [1, 1] #or [-1, -1]

# Assign the correct values to 'left_point', 'right_point' and 'plus_pos'.
### YOUR CODE
left_point[1] = -left_point[0]
right_point[1] = -right_point[0]
plus_pos = [1, 1]
### END CODE

# The following lines produce the plots. Try run the code.  
# See the documentation at https://matplotlib.org/contents.html
plt.plot([left_point[0], right_point[0]], [left_point[1], right_point[1]], 'k--', linewidth=4, label=r'$\theta^{T} x = 0$')
plt.plot(plus_pos[0], plus_pos[1],'r', marker='+', markersize = 16, linewidth=4)
plt.legend()
plt.show()

# ======================================== QUESTION 4 ========================================
# In python you can extract the i'th column by A[:, i-1] (notice that the columns are 0 indexed)
A = np.array([[-3, 6,6,6], [7,9,0,2], [42,0,3, -5], [0, 0, 7, 4]])
e3 = np.array([0,0,1,0])
e2 = np.array([0,1,0,0])

# Ae3 = A[:,2]
# Ae2 = A[:,1]


### YOUR CODE - FILL Ae3, Ae2, A4e3_2e2
Ae3 = np.dot(A,[0,0,1,0])
Ae2 = np.dot(A,[0,1,0,0])
A4e3_2e2 = 4*Ae3+2*Ae2
### END CODE

print('Ae3: ', Ae3)
print('Ae2: ', Ae2)
print('A(4e3+2e2): ', A4e3_2e2)


# ======================================== QUESTION 5 ========================================


# In python ** is exponentiation i.e. 2**10 = 1024
print('Two to the power of 10: ', 2**10)

### YOUR CODE HERE assign values to p_hthht and p_hhhhh
p_hthht = 0.5**5
p_hhhhh = 0.5**5
### END CODE 

print('Probability of heads, tails, heads, heads, tails: ', p_hthht)
print('Probability of heads, heads, heads, heads, heads: ', p_hhhhh)

# ======================================== QUESTION 6 ========================================

# Write the gradient in the lambda function below (instead of None).
f = lambda x: (0.5*x+2)**2
nablaf = lambda x: 0.5*x+2
hessian = lambda x: 0.5

### YOUR CODE HERE
### END CODE

import inspect
print('nablaf: ', inspect.getsource(nablaf))
print('hessian: ', inspect.getsource(hessian))
z = 1.0/2.0
nabla_f_z = nablaf(z)
print("f'(1/2): ", nabla_f_z)
xs = np.linspace(-3, 3, 300)
plt.plot(xs, [f(y) for y in xs], 'b-', label='f(x)')
tangent_x = [z-1, z+1]
tangent_y = [f(z) - nabla_f_z, f(z) + nabla_f_z]
plt.plot(tangent_x, tangent_y, 'r-', label='Derivative at {0}'.format(z))
plt.legend()
# plt.show() # UNCOMMENT TO SHOW TANGENT FIG

# ======================================== QUESTION 7 ========================================

# Write the Jacobian in the lambda function below
# Remember that pow is ** in python
nablaf = lambda x: (x[1]+2*x[0]*x[1], x[0]+x[0]**2)

### YOUR CODE HERE
### END CODE

x = np.array([2.0, 3.0])
print('nabla f(2,3): ', nablaf(x))

# ======================================== QUESTION 8 ========================================
# Write the expected value as a function of n in the lambda function below
en = lambda n: sum(np.arange(1,n+1))/n

### YOUR CODE HERE

### END CODE

print('for n = 10: teachers guess 5.5: ', en(10))
print('for n = 2: teachers guess 1.5: ', en(2))

# ======================================== QUESTION 9 ========================================
# This may be useful
# np.infty?

### YOUR CODE HERE
max_children = np.infty
exp_children_n = lambda n: sum(list(i*0.5**i for i in np.arange(1,n+1)))
exp_children = exp_children_n(100) # 100 is some high number
### END CODE

print('Max Number of children: ', max_children)
print('Expected Children is: ', exp_children)


# ======================================== QUESTION 10 ========================================
>
