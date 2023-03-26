#!/usr/bin/env python
# coding: utf-8

# # An Introductory Tour of Python for CS 3630

# ## Google Colab & ipynb Introduction
# 
# ### What is an ipynb file?
# An ipynb file is a notebook document created by Jupyter Notebook, an interactive computational environment that was created to help data science & ML professionals use Python to interact with their data. A .py file is a regular, plain text python file (that just contains code), whereas a .ipynb file is a python notebook that contains your code, results, and any other text or information.
# ### What is Google Colab?
# Google Colab is a tool that allows you to write and run python code in your browser itself, with no configuration required! It even allows you to save your precious Colab Files (which are .ipynb files) in Google Drive. The neat thing about Colab is that along with your executable
# python code and text, you can put in images, LaTeX, or HTML, making it incredibly useful for both teaching and learning (especially Robotics & Perception!).
# ### How do we run an ipynb file in Google Colab?
# In a .ipynb file, you can have both code and text. Each snippet of code or text is separated into cells, and each cell typically serves a specific purpose. For example, one cell can be for imports (numpy, pandas, etc), the next cell can be for defining any variables, and the following cell can be for computations. In between each of those you can even have cells of text that explain what you are doing, like commenting. Each individual cell can be run using the play button on the left of the cell, or using “Command/Ctrl+Enter”. To run all of the cells, you can click the Runtime button on the Toolbar and click “Run All.”
# ### What are the instructions for submitting a file?
# In order to use a file as your own, once we give you the colab notebook link:
# 1. Click the “File” button on the toolbar at the top
# 2. Click “Save a Copy”
# 3. Work on the project within the copy (it will say “Copy of…”)
# In order to submit a file, once you complete the project:
# 1. Click the “File” button on the toolbar at the top
# 2. Click “Download,”
# 3. And then click “Download .py”
# 4. You will now have the .py file on your local machine. Rename the file to 'submission.py' and submit the file to gradescope
# 
# ### More resources!
# Pretty much all the basics related to Colab can be found at colab.research.google.com. It includes a video tutorial, some cells to play around with, and some more information on Colab.

# ## Basic Python
# 
# Welcome to CS 3630. We hope you're as excited for this class as we are excited to teach it.
# 
# In this assignment, we introduce some basic tools and techniques which will hopefully be useful to you as you tackle all the labs this semester, as well as ensure that your development environment is set up correctly.
# 
# Some of you may already be familiar with all the topics in this assignment, so think of this assignment as more of a "working out the initial gremlins".

# In[1]:


get_ipython().system('pip install gtsam')
import numpy as np
import gtsam


# ### Printing
# 
# The most fundamental aspect of any programming language is being able to print out values, which is incredibly easy in Python.
# 
# Being able to print with nice formatting is particularly helpful in debugging.

# In[2]:


print("Welcome to CS 3630")

term = "Spring"
year = 2023
print("This is for {0} {1}".format(term, year))


# One powerful way to print lots of variables conveniently is to use `f-strings`. `f-strings` stand for format-strings which leverage special syntax for formatting. You specify an `f-string` by prepending the string with an `f` and printing variables in it by specifying the variable name with curly braces, e.g. `{var}`.

# In[3]:


name = "Hardik Goel" #TODO Add your name here.
fstring =f"This is a f-string to display your name: {name}"
print(fstring)


# ### Looping
# 
# A lot of operations involve for-loops, which in python is very idiomatic.
# It is very similar to the syntax in Java, however Python requires ':' instead of '{' after the if-else condition. An indentation denotes the scope of the statement, and nothing is required after the indented block of code. 

# In[4]:


x = [1, 3, 5, 7]

vanilla_sum = 0
for i in range(len(x)):
    vanilla_sum += x[i]

print(vanilla_sum)


# However, you may want to use the `enumerate` built-in method for most of your loops as it lets you iterate through elements while also providing an index into them.
# 
# Why don't you try adding the line of code to do that? It should be very similar to the above loop statement.

# In[5]:


sum = 0
for index, value in enumerate(x):
    sum += value #TODO Add code to sum the values in x here
assert sum == vanilla_sum
print(sum)


# Things like `if-else` statememts, classes, and functions are pretty much the same as any other programming language you've used in the past, so we won't spend too much time on those.
# To learn more about these topics, visit: [Python tutorial](https://www.tutorialspoint.com/python/index.htm). The topics are divided into sections on the left of the site. 

# ### Comprehensions & Lambdas
# 
# List comprehensions are a powerful tool to make your code more succinct and easier to read.
# 
# Say you want to filter out all the even numbers from a list into a new list, this is easily done with a list comprehension.

# In[6]:


x = list(range(20))

evens = [c for c in x if c % 2 == 0]
print(evens) 


# Let's see if you can generate the first 6 Mersenne numbers using a list comprehension. A Mersenne number is a number that is of the form $M_n = 2^n - 1$.
# 
# **NOTE** 0 is not a Mersenne number (since its binary representation is not all 1s), so be sure to adjust your loop range accordingly.

# In[7]:


#TODO Add a list comprehension here to generate the first 6 Mersenne numbers.

mersennes = [((2**c) - 1) for c in range(1, 7)]
assert mersennes == [1, 3, 7, 15, 31, 63], f"Your code produced {mersennes}"


# ## Numpy

# ### Vectors & Matrices
# 
# The most fundamental data structure in python is an `ndarray` which is short for n-dimensional array (also called a `tensor`). You can easily make any list/tuple into an `ndarray` using the following two methods: `np.array` & `np.asarray`. The main difference between the two is that 'np.array' makes a copy of the input by default whereas `np.asarray` does not. Changes to the result of `np.array` will therefore only be applied to this copy and not the original array object.

# In[8]:


x = list(range(10))

y = np.array(x)
z = np.asarray(x)

print(y)
print(z)
# We can assert that they are truly equal.
assert np.all(y == z)


# Most of the time, you'll want to create a matrix or vector that is all zeros or ones. The methods for these are `np.zeros` & `np.ones`, both of which accept the expected shape. The shape may be provided as a scalar value or a tuple.

# In[9]:


zero_vector = np.zeros(10)
print(zero_vector)
assert zero_vector.shape == (10,)

ones_matrix = np.ones((2, 5))
print(ones_matrix)
assert ones_matrix.shape == (2, 5)

#TODO Create a zeros matrix of size 10x4
zero_points = np.zeros((10, 4))
assert zero_points.shape == (10,4) and np.all(zero_points == 0), f"Incorrect shape {zero_points.shape} or has non-zero elements"

#TODO  Create a ones matrix of size 3x12
one_stack = np.ones((3, 12))
assert one_stack.shape == (3,12) and np.all(one_stack == 1), f"Incorrect shape {one_stack.shape} or has non-one elements"


# You may need some structured vectors or matrices, such as a sequence vector or the identity matrix. Those are simple as well.

# In[10]:


seq = np.arange(1, 10)  # Vector from 1-9

I_3x3 = np.eye(3)  # A 3x3 identity matrix
#TODO Create a 5x5 identity matrix
I_5x5 = np.eye(5, 5)
assert I_5x5.shape == (5,5) and np.all(np.diag(I_5x5) == 1) and np.all(I_5x5[~np.eye(5, dtype=bool)] == 0), f"I_5x5 is not a valid identity matrix."


# ### Indexing
# 
# Indexing in `numpy` is slightly different from indexing in traditional python. In `numpy`, you specify all index ranges within a single pair of box brackets. The range semantics (e.g. `[0:10]`) are still the same though.
# 
# For example, to index a 2D matrix, use comma-separated values within square brackets (i.e., `X[1,2]`). To index a range of values in this 2D matrix, we use the `:` (i.e., `X[1:3,2:6]`). Note that the final index is not inclusive. Additionally, the first or last value can be dropped to remove redundancy in the event that they are 0 or length of that dimension, respectively.

# In[11]:


I_6x6 = np.eye(6)

# Get the bottom-right 3x3 submatrix
I_3x3 = I_6x6[3:6, 3:6]

# Asserts that the bottom-right submatrix is a 3x3 identity
assert np.all(I_3x3 == np.eye(3))

X = np.asarray(I_6x6)
X[5, 0] = 13
X[4, 1] = 12
X[1, 4] = 11
X[0, 5] = 10

#TODO Get the top right 3*3 submatrix of X
top_right = X[0:3, 3:6]
assert np.all(top_right == np.asarray([[0, 0, 10], [0, 11, 0], [0, 0, 0]])), f"Incorrect submatrix indexing"


# ### Shapes & Broadcasting
# 
# Many times, just knowing the shape of your matrix can help in figuring out what the correct operation should be. This is easily achievable with the `np.shape(x)` method (which is also an attribute of any numpy array).

# In[12]:


x = np.empty((5, 12))
#TODO Call 'shape' on x to get the correct value
shape = np.shape(x)
assert shape == (5, 12), f"Incorrect shape received, expected (5, 12)"


# Once you know the current shape, sometimes you may want to add a single (or maybe multiple) dimension. This is particularly common in `numpy` where vectors are represented as shape `(N,)` and are different from shape `(N,1)` (which is a Nx1 matrix), even though they both have the same number of elements.
# 
# Sometimes you want to work with the latter, so you can reshape it easily:

# In[13]:


N = 8
x = np.arange(N)
assert x.shape == (N,)

x1 = x.reshape((N, 1))
assert x1.shape == (N, 1)

#TODO reshape x to (1, N)
x2 = x.reshape((1, N))
assert x2.shape == (1, N), f"x2 has shape {x2.shape}, expected (1, 8)"


# Other times, you just want to add a dimension for a quick operation, such as matrix-matrix multiplication. You can add new dimensions by indexing with `np.newaxis` or, more conveniently, `None`.

# In[14]:


x = np.arange(10)
x1 = x[:, None]

assert x1.shape == (10, 1)

y = np.arange(20)
y11 = y[None, :, None, None] #TODO Add three dimensions, one before and two after the main vector dimension
assert y11.shape == (1, 20, 1, 1), f"Incorrect dimensions {y11.shape}, should be (1, 20, 1, 1)"


# ### Arithmetic Operations
# 
# In general, prefer performing arithmetic on arrays using __vectorized__ code over `for` loops because numpy vectorized arithmetic is orders of magnitude faster.

# In[15]:


# Don't worry about understanding this block of code; it's just to measure the execution speed.
from time import perf_counter
from contextlib import contextmanager
@contextmanager
def measure_time(msg) -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f'{msg:10s} took {perf_counter() - start:.6f} seconds')


# In[16]:


x = np.arange(1e5)
y = np.arange(2e5, 3e5)

# Bad
with measure_time('Bad') as _:
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    z = np.array(z)

# Still bad
with measure_time('Still Bad') as _:
    z = np.zeros(x.shape)
    for i, (xi, yi) in enumerate(zip(x, y)):
        z[i] = xi + yi

# Best
with measure_time('Best') as _:
    z = x + y


# ###Matrix Multiplication
# 
# Most matrix/vector arithmetic is element-wise, with one important exception being general matrix multiplication (commonly referred to as 'matmul' or 'gemm'), which is defined using the `@` operator in `numpy`.  Matmul is the workhorse of linear algebra and, by consequence, robotics. **The '*' operator does not denote matrix multiplication, but is actually the operator for an element-wise multiplication.**
# 
# Recall from linear algebra that you can find the sum of elements in a 1D matrix by matrix-multiplying it with a matrix of ones in a transposed shape:
# 
# $${\begin{bmatrix}
# 1 & 2 & 3
# \end{bmatrix}}
# {\begin{bmatrix}
# 1\\1\\1
# \end{bmatrix}}=6$$

# In[17]:


x = np.asarray([5., 10, 15])
y = np.asarray([1., 2, 3])
I_3x3 = np.eye(3)
print(f'x = {x}')
print(f'y = {y}')
print(f'x + y = {x + y}')
print(f'x - y = {x - y}')
print(f'x * y = {x * y}')
print(f'x / y = {x / y}')
print(f'I_3x3 @ x = {I_3x3 @ x}')
print(f'np.sum(x) = {np.sum(x)}')
print(f'np.prod(x) = {np.prod(x)}')

O = np.ones((1, 3))
print(f'O = {O}')
sum_x = np.dot(O, x)  # TODO: Without using `sum` but instead with matrix multiplication using `O`, calculate the sum of the elements in x.
assert sum_x == np.sum(x), f"You computed sum_x = {sum_x}, but should be {np.sum(x)}"


# ### Broadcasting
# 
# Broadcasting is one the most powerful aspects of `numpy`, but is also tricky to understand. The best way to understand it is to try to use it as much as possible and gain an intuitive feeling for it.
# 
# This feature allows you to specify matrices/vectors of different sizes (albeit with some conditions) and operate on them together. A common example is element-wise matrix vector product as shown below. Be sure to checkout the [numpy broadcasting basics page](https://numpy.org/doc/stable/user/basics.broadcasting.html) for more details!

# In[18]:


X = np.array([[0, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 3],
       [0, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 3],
       [0, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 3]])

assert X.shape == (12, 4)
scale = np.arange(6, 10)  # shape is (4,)

# Broadcasting scale (4,) to (12, 4) for element-wise multiplication
Y = X * scale

assert np.all(Y == np.tile(np.diag([0, 7, 16, 27]), 3).T), f"Scaled values are incorrect"


# ### Random Sampling
# 
# To finish up, we will look at examples of sampling from probability distributions commonly employed when you wish to generate random numbers.

# #### Generate a matrix of random values
# 
# Sometimes you just need some random values sampled uniformly from `[0, 1]`. This is easy to do with `np.random.default_rng` which follows the same semantics as `np.ones` and creates a random number generator. Using this generator, 'random' can be used to generate random values sampled uniformly from `[0, 1]`.

# In[19]:


rng = np.random.default_rng()
R = rng.random((3, 3))
print(R)


# #### Samples from the Normal Distribution
# 
# The Gaussian/Normal distribution is the most common probability distribution used in robotics due to it having some very nice properties, one of which is ease of sampling.
# 
# We can sample from both a standard Gaussian, as well as a custom Gaussian using the same random number generator.

# In[20]:


std_samples = rng.standard_normal((3, 3))
print(std_samples)

# The first argument is the mean, the second is the standard deviation
# We specify the standard deviation to be 1.0 which is pretty tight around the mean.
# This means that the samples won't be too far away from the mean value aka 5
custom_gaussian = rng.normal(5, 1.0, (3, 3))
print(custom_gaussian)


# #### Sampling choices
# 
# Finally, one common scenario is having to choose from a set of options, e.g. should you move or not move, should you pick object A, B or C, etc.
# This is where the `choice` method is useful since it allows you to sample discrete values easily.

# In[21]:


# Give us 6 random choices from 0-9 without replacement
choices = rng.choice(10, size=6, replace=False)
print("Choices without replacement:", choices)

# Give us 7 random choices from 20-29 with replacement
choices = rng.choice(np.arange(20, 30), size=7, replace=True)
#NOTE you may or may not see duplicates, because it is (pseudo-)random!
print("Choices with replacement:", choices)

# Sample random actions for the robot
actions = ['move forward', 'move right', 'move left', 'move backward']
samples = rng.choice(actions, size=3, replace=True)
print(samples)


# ### Conclusion
# 
# Hopefully you find this quick tour of `python` and `numpy` informative and useful. A lot of the tips and tricks here should serve you well during the course and beyond it, making you both a very competent Python programmer as well as a roboticist.
# 
# Looking forward to having you in the class!
