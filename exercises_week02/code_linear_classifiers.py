import os 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import urllib

def load_data():
    filename = "week2_exercise_data.npz"
    if not os.path.exists(filename):
        # os.system('wget https://users-cs.au.dk/jallan/ml/data/{0}'.format(filename))
        with open(filename,'wb') as fh:
            fh.write(urllib.request.urlopen("https://users-cs.au.dk/jallan/ml/data/%s" % filename).read())            
    D = np.load(filename)
    return D

def plot_hyperplane_example():
    """  
    Plot the hyperplane represented by w = [−1.0, 1.0, 1.0] in the range [0,1]x[0,1]. 
    Approach: 
    Find the x2-coordinate for the hyperplane (line) at x1=0 and at x1=1.
    You need to solve the hyperplane equations to do it: w^T * x = w0*1 + w1*x1 + w2*x2 = 0 
    If you know all but one variable you should be able to solve it.
    
    Write code here to create two NumPy arrays x and y that contain the x and y coordinates of the two endpoints of the line to be drawn.
    """
    x = np.array((0,0))
    y = np.array((0,0))

    ### YOUR CODE 2 lines
    ### END CODE 
    
    plt.axis([0, 1, 0, 1])
    plt.plot(x,y)
    plt.title('hyperplane w={0}'.format([-1, 1, 1]))
    plt.show()
    
def plot_hyperplane(w, *args, **kwargs):
    """ 
    Plot the hyperplane (line) w0 + w1*x1 + w2*x2 = 0 in the range R = [xmin,xmax] times [ymin,ymax] for a generic w = (w0,w1,w2).
    
    We will proceed in a similar fashion as we did in the previous task.
    There we had xmin = ymin = 0 and xmax = ymax = 1 (i.e. the range was [0,1]x[0,1]), and we just found the intersection points of the hyperplane with the two vertical lines x1=0 and x1=1. 
    How can we find these two points for a generic w = (w0,w1,w2) and a generic range ([xmin,xmax] x [ymin,ymax])?
 
    Remember to handle possible special cases! 
    
    Notice how we pass along optional arguments to the plot function, which allows us to change color, etc. of the hyperplanes.

    Args:
    w: numpy array shape (d,)
    args: extra arguments to plot (ignore)
    kwargs: extra keyword arguments to plot (ignore)
    """
    
    if w[1]==0 and w[2]==0: raise ValueError('Invalid hyperplane')
    # Notice that w1 and w2 are not allowed to be 0 simultaneously, but it may be the case that one of them equals 0
    
    xmin, xmax, ymin, ymax = plt.axis()
    
    # Write the code here to create two NumPy arrays called x and y.
    # The arrays x and y will contain the x1's and x2's coordinates of the two endpoints of the line, respectively.
    
    x = np.array((0,1))
    y = np.array((0,1))
    
    ### YOUR CODE
    ### END CODE
    
    # plot the line
    plt.plot(x, y, *args, **kwargs)


def pla_train(X, y, w=None):
    """
    Perceptron learning algorithm
    
    The Perceptron algorithm finds a hyperplane represented by a parameter (normal, i.e. it has norm 1) vector wpla that splits the input
    domain such that all points with +1 labels is on one side of the hyperplane and the points with labels -1 are on the other (if that is possible). 
    
    To label a new point x, we simple output sign(wpla^T * x)

    The functions np.dot, np.sign, np.nonzero, np.random.choice may come in useful.
    Also, recall that NumPy allows us to perform the same operation on each entry of an array easily.
    For example, if A is an array, A+2 is the array obtained by adding 2 to each entry of A (we could have used any other 'operator' instead of +)
    
    Args:
    X: numpy array shape (n,d), each row represents one vector of the given data
    y: numpy array shape (n,), these are the corresponding labels
    w: numpy array shape (d,), it's the starting vector for the algorithm
    
    Returns:
    w: numpy array shape (d,) normal vector of a hyperplane separating X, y
    """
    
    # if not starting vector is given we set it to the zero vector
    if w is None:
        w = np.zeros(X.shape[1])
    
    # Run the perceptron iteration
    
    ### YOUR CODE 5-10 lines
    ### END CODE   
        
    return w

def lin_reg_train(X, y):
    """ 
    Linear Regression Learning Algorithm

    Linear regression computes the linear model (line in 2D) that
    best approximates the real valued target.
    For this we compute the parameter vector 
    
    wlin = argmin ( sum_i (w^T x_i -y_i)^2 )
    
    The pseudo-inverse operator pinv in numpy.linalg package may be useful, i.e. np.linalg.pinv

    Args:
    X: numpy array shape (n,d)
    y: numpy array shape (n,)
    
    Returns:
    w: numpy array shape (d,) the best weight vector w to linearly approximate the target from the features.

    """  
    w = np.zeros(X.shape[1])
    #YOUR CODE 1-3 lines
    w = np.dot(np.linalg.pinv(X),y)
    #END CODE
    return w

def plot_result(dat, target, w_lin, w_pla, title):
    """
    Compute in sample error and plot decision boundaries
    The in-sample error is the number of mispredictions of each classifier (perceptron, linear regression)
    
    p_in = \sum_{x,y in dat, target} 1_{if f(x) != y else 0}
    
    The functions np.sum, np.dot, np.sign may be useful
    
    Args:
    dat: np.array shape (n,d)
    target: np.array shape (d,)
    w_lin: np.array shape (d,)
    w_pla: np.array shape (d,)
    title: string
    """
    
    plt.figure(figsize=(10,8))
    plt.axis([dat[:, 1].min(), dat[:, 1].max(), dat[:, 2].min(), dat[:, 2].max()])
    
    #Compute In Sample Error for the two models and store in p_in, l_in 
    p_in = 0
    l_in = 0
    
    ### YOUR CODE 2-4 lines
    ### END CODE
    
    print(title, '- In Sample Error:\n\tPerceptron: {0}\n\tLinear Regression: {1}'.format(p_in, l_in))
    # plot data
    plt.scatter(dat[:,1], dat[:,2], c=target, cmap=plt.cm.Paired, s=6)
    #plot pla line
    plot_hyperplane(w_pla, 'r--', linewidth=4, label='perceptron')    
    #plot lin reg line
    plot_hyperplane(w_lin, 'g--', linewidth=4, label='lin. reg.')    
    print('w_pla vector', w_pla)
    print('w_lin vector', w_lin)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig('{0}.png'.format(title))

def test_hyperplane():
    plt.axis([-1, 1, -1, 1])
    plot_hyperplane((1, 2, 0), 'r--', linewidth=4)
    plot_hyperplane((1, 0.75, -3), 'g--', linewidth=4)
    plt.show()

    
def run():
    D = load_data()
    print(D.files)
    dat = D
    dsets = [(dat['dat%d' % i], dat['target%d' % i]) for i in range(1, 5)]
    for i,(X,y) in enumerate(dsets):
        w_pla = pla_train(X, y, np.zeros(3))
        w_lin = lin_reg_train(X, y)
        plot_result(X, y, w_lin, w_pla, "Data Set {0}".format(i+1))
    plt.show()
    print('Can you explain the results?')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-simple', action='store_true', default=False)
    parser.add_argument('-hyperplane', action='store_true', default=False)
    parser.add_argument('-run', action='store_true', default=False)
    args = parser.parse_args()
    if args.simple:
        plot_hyperplane_example()
    if args.hyperplane:
        test_hyperplane()
    if args.run:
        run()

