print('Compute the outer product first by hand, then check with np.outer.')
print('You can use np.dot as well if you make \'x\' and \'theta\' into matrices of shape 3,1 with np.expand_dims\n')

# Make a variable called 'outer_prod' and assign it the outer product of 'x' and 'theta'.
outer_prod = np.outer_pr
### YOUR CODE HERE
### END CODE

# Prints the outer product and its shape. 
print('Outer Product Shape Is', outer_prod.shape)
print('Outer Prodcut Is:\n', outer_prod)

print('\nUsing np.expand_dims:')
print(np.expand_dims(x, axis=1) @ np.expand_dims(theta, axis=1).T) #The @ symbol is the matrix multiplication operator
