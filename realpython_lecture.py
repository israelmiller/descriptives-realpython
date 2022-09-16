#import packages
import math
from re import X
import statistics
from tkinter import Y
import numpy as np
import scipy.stats
import pandas as pd

#Create a list of numbers

x = [8.0, 1, 2.5, 4, 28.0]

x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]

#display the lists
x
x_with_nan

###
#You can see that the functions are all equivalent.
# However, please keep in mind that comparing two nan values for equality returns False. 
# In other words, math.nan == math.nan is False!
###

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

y 
y_with_nan

z
z_with_nan

##Mean
#built-in, no imports needed
mean_ = sum(x) / len(x)
mean_

#you can also use the statistics module
mean_ = statistics.fmean(x)
mean_
#you can use mean or fmean, but fmean is faster. fmean also always returns a float, even if the input is an integer.
mean_ = statistics.mean(x)
mean_

#if there is a nan value statistics.mean and statistics.fmean will return a nan value
mean_ = statistics.mean(x_with_nan)
mean_
mean_ = statistics.fmean(x_with_nan)
mean_
#this is consistent with sum which will return a nan value if there is a nan value in the list

#you can use numpy to calculate the mean
mean_ = np.mean(y)
mean_
#you can also use numpy mean as a method
mean_ = y.mean()
mean_

#numpy mean also returns nan if there is a nan value in the array
mean_ = np.mean(y_with_nan)
mean_

#you can use numpy nanmean to calculate the mean ignoring nan values
mean_ = np.nanmean(y_with_nan)
mean_

#the pandas mean method ignores nan values by default
mean_ = z_with_nan.mean()
mean_

##Weighted Mean
# a generalization of the arithmetic mean that enables you to define the 
# relative contribution of each data point to the result.
# the weights are usually positive and sum to 1
# you can implement the weighted mean by combining sum() with eith range() or zip()

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]

wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

# you can also use NumPY to calculate the weighted mean with np.average()
wmean = np.average(x, weights=w)
wmean

#be careful if the dataset contains nan values as np.average() will return nan
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
wmean = np.average(x_with_nan, weights=w)
wmean

##Harmonic Mean
# the harmonic mean is the reciprocal of the mean of the reciprocals of all items in the dataset.
hmean = len(x) / sum(1 / i for i in x)
hmean

#it can also be calculated with the statistics module using the harmonic_mean() function
hmean = statistics.harmonic_mean(x)
hmean

#if you have a nan value in the dataset, the harmonic_mean() function will return a nan value
hmean = statistics.harmonic_mean(x_with_nan)
hmean
#if there is at least one 0 in the dataset, the harmonic_mean() function will return 0
statistics.harmonic_mean([0, 1, 2, 3, 4, 5])

#if you provide a negative value to the harmonic_mean() function, it will raise a ValueError
statistics.harmonic_mean([-1, 1, 2, 3, 4, 5])

#you can also use scipy.stats.hmean() to calculate the harmonic mean
scipy.stats.hmean(y)
scipy.stats.hmean(z)

##Geometric Mean
# the geometric mean is the nth root of the product of all items in the dataset.
#you can implement geometric mean in pure Python:
gmean = 1
for i in x:
    gmean *= i
gmean **= 1 / len(x)
gmean

#python also has a geometric_mean() function in the statistics module
gmean = statistics.geometric_mean(x)
gmean
#if there is a nan value in the dataset, the geometric_mean() function will return a nan value
gmean = statistics.geometric_mean(x_with_nan)
gmean

#you can also use scipy.stats.gmean() to calculate the geometric mean
scipy.stats.gmean(y)
scipy.stats.gmean(z)

##Median
# the median is the middle value in a dataset when the values are sorted in ascending order.

#you can implement the median in pure Python
n = len(x)
if n%2:
    median_ = sorted(x)[round(.5*(n-1))]
else:
    x_ord, index = sorted(x), round(.5*n)
    median_ = .5*(x_ord[index-1] + x_ord[index])

median_

#you can also use the statistics module to calculate the median
median_ = statistics.median(x)
median_

# if there is an even number of items in the dataset, the median() function will return the mean of the two middle values
median_ = statistics.median(x[:-1])
median_

# you can use median_low() and median_high() to get the lower and upper median values
statistics.median_low(x[:-1])

statistics.median_high(x[:-1])

#the median functions in the statistics module will not return nan if there is a nan value in the dataset
median_ = statistics.median(x_with_nan)
median_

# you can also use numpy to calculate the median
median_ = np.median(y)
median_

#numpy will return an error if there is a nan value in the dataset
median_ = np.median(y_with_nan)
median_
# you can use numpy nanmedian to calculate the median ignoring nan values
median_ = np.nanmedian(y_with_nan)
median_

##Mode
# the mode is the most common value in a dataset.
# you can implement the mode in pure Python
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

# you can also use the statistics module to calculate the mode
mode_ = statistics.mode(u)
mode_

mode_ = statistics.multimode(u)
mode_

# the mode() function will return an error if the dataset is multimodal
u = [2, 3, 2, 8, 12, 8, 8, 2]
#mode_ = statistics.mode(u)
#mode_
#I guess above is no longer true? I did not get an error

mode_ = statistics.multimode(u)
mode_

#statistics.mode and statistics.multimode will return a nan value if there is a nan value in the dataset
u_with_nan = [2, 3, 2, 8, 12, 8, 8, 2, np.nan, np.nan, np.nan, np.nan]
mode_ = statistics.mode(u_with_nan)
mode_

###Measures of Variability
## measures of variability are used to describe the spread of the data in a dataset.

##Variance
#Quantifies the spread of the data in a dataset.
