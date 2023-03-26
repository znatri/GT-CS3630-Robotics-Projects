#!/usr/bin/env python
# coding: utf-8

# # CS3630 Project 1: Trash Sorting Robot (Spring 2022)
# ## Brief
# - Due: Mon, Jan 30 at 11:59pm on gradescope
# - Hand-in: through Gradescope
# 
# ## Getting started
# In order to use a file as your own, once we give you the notebook link:
# 1. Download the file to your computer
# 2. Upload the notebook to Google Colab (File > Upload Notebook)
# 
# ## Submission Instructions
# In order to submit a file, once you complete the project:
# 1. Click the “File” button on the toolbar at the top
# 2. Click “Download,”
# 3. And then click “Download .py”
# 4. You will now have the .py file on your local machine.
# 5. Make sure it is named `project1.py`
# 6. Submit the `project1.py` file to gradescope
# 
# 
# ## Introduction
# Welcome to your first project in CS3630 (Spring 2022)!
# 
# In this project, we will be building a (simulated) trash sorting robot as illustrated in the [textbook](http://www.roboticsbook.org/intro.html) for this course. In this scenario, the robot tries to sort trash of some pre-determined categories into corresponding bins. Please refer to [Chapter 2](http://www.roboticsbook.org/S20_sorter_intro.html) of the book for a more detailed description of the scenario. **This project is basically based on Chapter 2 of the textbook. Please use the same values in the textbook for each TODO.**

# First, install gtsam and import some other useful libraries.

# In[3]:


# To use on colab, run the following line
get_ipython().system('pip install -U -q gtbook')


# In[4]:


#export
import gtsam
import numpy as np
import math
from enum import Enum
from gtbook.discrete import Variables


# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


# Download the project1_test file to check your code on colab
get_ipython().system(' pip install --upgrade --no-cache-dir gdown')
get_ipython().system(' gdown --id 1m6K4c0njbAxHlBGM8FpDYYvdNDNdaxIA')


# In[7]:


from project1_test import TestProject1
from project1_test import verify


# In[8]:


np.random.seed(3630)
unit_test = TestProject1()


# **IMPORTANT NOTE: Please use the variables provided for the results of each of the TODOs.**
# ## Modeling the World State ([Book Section 2.1](http://www.roboticsbook.org/S21_sorter_state.html))
# - Functions to complete: **TODO 1**, **TODO 2**, and **TODO 3**
# - Objective: Representing the prior probabilities of the trash categories and simulate it by sampling. Please use the prior probabilities provided in the textbook

# In[9]:


#export
### ENUMS ###
class Trash(Enum):
    CARDBOARD = 0
    PAPER = 1
    CAN = 2
    SCRAP_METAL = 3
    BOTTLE = 4


class Bin(Enum):
    GLASS_BIN = 0
    METAL_BIN = 1
    PAPER_BIN = 2
    NOP = 3


class Detection(Enum):
    BOTTLE = 0
    CARDBOARD = 1
    PAPER = 2


### CONSTANTS ###
# All possible trash categories
CATEGORIES = ['cardboard', 'paper', 'can', 'scrap_metal', 'bottle']

# All possible actions/bins (nop means no action)
ACTIONS = ['glass_bin', 'metal_bin', 'paper_bin', 'nop']


# Useful Global Variables
variables = Variables()
categories = CATEGORIES
Category = variables.discrete('Category', categories)
Conductivity = variables.binary('Conductivity')
Detection = variables.discrete('Detection', ['bottle', 'cardboard', 'paper'])


# **TODO 1 & TODO 2**:

# In[10]:


#export
# TODO 1:
# Prior probabilities
def get_category_prior():
    '''
    Returns the prior probabilities of the trash categories.

        Parameters:
            None

        Returns:
            category_prior (gtsam.DiscreteDistribution): a DiscreteDistribution
                that summarizes the prior probabilities of all trash categories
    '''
    category_prior = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    category_prior = gtsam.DiscreteDistribution(Category, "200/300/250/200/50")
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return category_prior


# TODO 2:
# Prior probabilities PMF
def get_category_prior_pmf():
    '''
    Returns the probability mass function (PMF) of the prior probabilities
    of the trash categories.

        Parameters:
            None

        Returns:
            category_prior_pmf (list): a list of the PMF
    '''
    category_prior_pmf = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    category_prior = get_category_prior()
    category_prior_pmf = category_prior.pmf()
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return category_prior_pmf


# In[11]:


print("Testing your prior probabilities of the trash categories: ")
print(verify(unit_test.test_get_category_prior_pmf, get_category_prior_pmf))


# **TODO 3**:

# In[12]:


#export
# TODO 3:
def sampling(cdf):
    '''
    Returns the sample according to the CDF, returning the integer index of the sampled value.

        Parameters:
            cdf (NDArray): Cumulative distribution function values for categories

        Returns:
            category (int): an int indicating the sampled trash category
    '''
    u = np.random.rand()
    for category in range(5):
        if u < float(cdf[category]):
            return category

def sample_category():
    '''
    Returns a sample of trash category by sampling with the prior probabilities
    of the trash categories

        Parameters:
            None

        Returns:
            sample (int): an int indicating the sampled trash category, the
                int-category mapping is at the beginning of this notebook
    '''
    sample = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    # category_prior_pmf = get_category_prior_pmf()
    # cdf = np.cumsum(category_prior_pmf)
    # sample = sampling(cdf)
    
    category_prior = get_category_prior()
    sample = category_prior.sample()
    
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return sample


# In[13]:


print("Testing your sample of trash category: ", verify(unit_test.test_sample_category, sample_category))


# ## Actions for Sorting Trash ([Book Section 2.2](http://www.roboticsbook.org/S22_sorter_actions.html))
# - Functions to complete: **TODO 4**
# - Objective: Representing actions and their corresponding costs, please use the data provided in the textbook

# In[14]:


#export
# TODO 4:
'''
    Fill out the cost table with corresponding costs, where the rows correspond
    to ACTIONS and the columns correspond to CATEGORIES.
'''
COST_TABLE = None
###############################################################################
#                             START OF YOUR CODE                              #
###############################################################################
COST_TABLE = np.array([
    [2, 2, 4, 6, 0],
    [1, 1, 0, 0, 2],
    [0, 0, 5, 10, 3],
    [1, 1, 1, 1, 1]
])
###############################################################################
#                              END OF YOUR CODE                               #
###############################################################################


# ## Sensors for Sorting Trash ([Book Section 2.3](http://www.roboticsbook.org/S23_sorter_sensing.html))
# - Functions to complete: **TODO 5-7** , **TODO 8-10** 
# - Objective: Representing conditional probabilities of sensors and simulate them by sampling, please use the data provided in the textbook

# **TODO 5-8**:

# In[15]:


#export
# TODO 5:
# 1. Conductivity - binary sensor
def get_pCT():
    '''
    Returns P(Conductivity | Trash Category)

        Parameters:
            None

        Returns:
            pCT (gtsam.DiscreteConditional): a DiscreteConditional that
                indicates the conditinal probability of conductivity given
                the trash category
    '''
    pCT = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pCT = gtsam.DiscreteConditional(
        Conductivity, [Category], "99/1 99/1 10/90 15/85 95/5"
    )
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return pCT

# TODO 6:
# 2. Detection - multi-valued sensor
def get_pDT():
    '''
    Returns P(Detection | Trash Category)

        Parameters:
            None

        Returns:
            pDT (gtsam.DiscreteConditional): a DiscreteConditional that
                indicates the conditinal probability of camera detection
                given the trash category
    '''
    pDT = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pDT = gtsam.DiscreteConditional(
        Detection, [Category], "2/88/10 2/20/78 33/33/34 33/33/34 95/2/3"
    )
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return pDT

# TODO 7:
# 3. Weight - continuous-valued sensor
def get_pWT():
    '''
    Returns P(Weight | Trash Category)

        Parameters:
            None

        Returns:
            pWT (np.array): a numpy array of lists that consists of the means
                and standard deviations that define the weight distribution of each
                trash category

    '''
    pWT = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pWT = np.array([
        [20, 10], [5, 5], [15, 5], [150, 100], [300, 200]
    ])
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return pWT

# TODO 8:
def sample_conductivity(category=None):
    '''
    Returns a sample of conductivity using the conditional probability
    given the trash category.

        Parameters:
            category (int): an int indicating the trash category

        Returns:
            conductivity (int): an int indicating the conductivity, with
                0 being nonconductive and 1 being conductive
    '''
    conductivity = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pCT = get_pCT()
    conductivity = pCT.sample(category)
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return conductivity


# In[16]:


print("Testing your sample conductivity: ", verify(unit_test.test_sample_conductivity, sample_conductivity))


# **TODO 9**:

# In[17]:


#export
# TODO 9:
def sample_detection(category=None):
    '''
    Returns a sample of detection using the conditional probability given
    the trash category.

        Parameters:
            category (int): an int indicating the trash category

        Returns:
            detection (int): an int indicating the sampled detection, the
                int-detection mapping is at the beginning of this notebook
    '''
    detection = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pDT = get_pDT()
    detection = pDT.sample(category)
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return detection


# In[18]:


print("Testing your sample detection: ", verify(unit_test.test_sample_detection, sample_detection))


# **TODO 10**:

# In[19]:


#export
# TODO 10:
def sample_weight(category=None):
    '''
    Returns a sample of weight using the conditional probability given
    the trash category.

        Parameters:
            category (int): an int indicating the trash category

        Returns:
            weight (double): a double indicating the sampled weight
    '''
    weight = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pWT = get_pWT()
    weight = np.random.normal(*pWT[category])
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return weight


# In[20]:


print("Testing your sample weight: ", verify(unit_test.test_sample_weight, sample_weight))


# ## Perception ([Book Section 2.4](http://www.roboticsbook.org/S24_sorter_perception.html))
# - Functions to complete: **TODO 11-15** 
# - Objective: Calculating likelihoods using different methods given the observations from the world, please use the data provided in the textbook

# **TODO 11**:

# In[21]:


#export
# TODO 11:
def likelihood_no_sensors():
    '''
    Returns the likelihoods of all trash categories using only priors,
    aka no sensors.

        Parameters:
            None

        Returns:
            likelihoods (list): a list of likelihoods of each trash category
    '''
    likelihoods = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    likelihoods = get_category_prior_pmf()
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return likelihoods


# In[22]:


print("Testing your likelihoods with no sensors: ")
print(verify(unit_test.test_likelihood_no_sensor, likelihood_no_sensors))


# Helper function you can use in the following TODOs

# In[23]:


#export
### HELPER FUNCTIONS ###
def Gaussian(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)


# **TODO 12**:

# In[24]:


#export
# TODO 12:
def likelihood_given_weight(weight):
    '''
    Returns the likelihoods of all trash categories using only the weight
    sensor (no priors)

        Parameters:
            weight (double): a double indicating the weight of trash

        Returns:
            likelihoods (list): a list of likelihoods of each trash category
    '''
    likelihoods = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pWC = get_pWT()
    likelihoods = np.array([Gaussian(weight, *pWC[index]) for index in range(5)])
    
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return likelihoods


# In[25]:


print("Testing your likelihoods using only the weight sensor: ")
print(verify(unit_test.test_likelihood_given_weight, likelihood_given_weight))


# **TODO 13**:

# In[26]:


#export
# TODO 13:
def likelihood_given_detection(detection):
    '''
    Returns the likelihoods of all trash categories using only the detection
    sensor (no priors)

        Parameters:
            detection (int): an int indicating the sampled detection, the
                int-detection mapping is at the beginning of this notebook

        Returns:
            likelihoods (list): a list of likelihoods of each trash category
    '''
    likelihoods = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    pDT = get_pDT()
    likelihoods = [v for (k, v) in pDT.likelihood(detection).enumerate()]
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return likelihoods


# In[27]:


print("Testing your likelihoods using only the detection sensor: ")
print(verify(unit_test.test_likelihood_given_detection, likelihood_given_detection))


# **TODO 14**:

# In[28]:


#export
# TODO 14:
def bayes_given_weight(weight):
    '''
    Returns the posteriors of all trash categories by combining the weight
    sensor and the priors

        Parameters:
            weight (double): a double indicating the weight of the trash

        Returns:
            posteriors (list): a list of posterior probabilities of each trash category
    '''
    posteriors = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    category_prior = get_category_prior()
    likelihoods = likelihood_given_weight(weight)
    weights = gtsam.DecisionTreeFactor(Category, likelihoods)
    posteriors = gtsam.DiscreteDistribution(weights * category_prior).pmf()
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return posteriors


# In[29]:


print("Testing your posteriors with the weight sensor and priors: ")
print(verify(unit_test.test_bayes_given_weight, bayes_given_weight))


# **TODO 15**

# In[30]:


#export
# TODO 15:
# Bayes with three sensors
def bayes_given_three_sensors(conductivity, detection, weight):
    '''
    Returns the posteriors of all trash categories by combining all three
    sensors and the priors

        Parameters:
            conductivity (int): an int indicating the conductivity, with
                0 being nonconductive and 1 being conductive

            detection (int): an int indicating the sampled detection, the
                int-detection mapping is at the beginning of this notebook

            weight (double): a double indicating the weight of the trash

        Returns:
            posteriors (list): a list of posterior probabilities of each trash category
    '''
    posteriors = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    category_prior = get_category_prior()
    conductivity_factor = get_pCT().likelihood(conductivity)
    detection_factor = get_pDT().likelihood(detection)
    weight_factor = gtsam.DecisionTreeFactor(Category, likelihood_given_weight(weight))
    posteriors = gtsam.DiscreteDistribution(
        conductivity_factor * detection_factor * weight_factor * category_prior
    ).pmf()
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return posteriors


# In[31]:


print("Testing your posteriors giving all three sensors: ")
print(verify(unit_test.test_bayes_given_three_sensors, bayes_given_three_sensors))


# ## Decision Theory ([Book Section 2.5](http://www.roboticsbook.org/S25_sorter_decision_theory.html))
# - Functions to complete: **TODO 16** 
# - Objective: Incorporating the cost table with the perception to reach a final sorting decision

# **TODO 16**:

# In[32]:


#export
# TODO 16:
### DECISION ###
def make_decision(posteriors):
    '''
    Returns the decision made by the robot given the likelihoods/posteriors you calculated

        Parameters:
            posteriors (list): a list of posteriors of each trash category

        Returns:
            action (int): an int indicating the action taken by the robot, the
                int-action mapping is at the beginning of this notebook
    '''
    action = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    action = np.argmin(COST_TABLE @ posteriors)
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return action


# In[33]:


print("Testing the decision made by your robot: ")
print(verify(unit_test.test_make_decision, make_decision))


# In[34]:


unit_test.get_cost_table(COST_TABLE)
print("Testing your cost without sensors: ")
print(verify(unit_test.test_score_likelihood_no_sensor, likelihood_no_sensors, make_decision))
print("Testing your cost using the weight sensor:")
print(verify(unit_test.test_score_likelihood_given_weight, likelihood_given_weight, make_decision))
print("Testing your cost using the detection sensor:")
print(verify(unit_test.test_score_likelihood_given_detection, likelihood_given_detection, make_decision))
print("Testing your cost using with the weight sensor and priors:")
print(verify(unit_test.test_score_bayes_given_weight, bayes_given_weight, make_decision))
print("Testing your cost using all three sensors: ")
print(verify(unit_test.test_score_bayes_given_three_sensors, bayes_given_three_sensors, make_decision))


# ## Extra Credit: Learning ([Book Section 2.6](http://www.roboticsbook.org/S26_sorter_learning.html))
# A Gaussian distribution, also known as a normal distribution, is an inappropriate distribution to represent
# the weight of an item. This is because it has an infinite range and therefore sampling from it can produce
# a negative number, while an item cannot have a negative weight. A more commonly used distribution
# used to represent weight is the [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution) which can only contain positive real values. The book explains how to fit a gaussian distribution to a set of data. For extra credit, we would like you to implement a function. 
# - Functions to complete: **TODO 17** 
# - Objective: Fit a Log-Normal Distribution to a set of data
# - Hint: There is an estimation of parameters section on the wikipedia article

# **TODO 17**:

# In[35]:


#export
# TODO 17
def fit_log_normal(data):
    '''
    Returns mu, sigma for a log-normal distribution

        Parameters:
            data (list of floats): A list of positive floats that represent the weight of an item

        Returns:
            mu (float), sigma (float): The mu and sigma for a log-normal distribution
    '''
    mu = None
    sigma = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################
    data = np.log(data)
    mu = np.mean(data)
    sigma = np.std(data)   
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return mu, sigma


# In[36]:


print("Testing your log-normal distribution: ", verify(unit_test.test_fit_log_normal, fit_log_normal))

