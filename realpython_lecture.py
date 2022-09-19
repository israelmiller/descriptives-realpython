#import packages
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

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

#you can implement the variance in pure Python
n = len(x)
mean_ = sum(x) / n
var_ = sum((item-mean_)**2 for item in x)/(n-1)
var_

#you can also use the statistics module to calculate the variance
var_ = statistics.variance(x)
var_
#nan values in the dataset will result in a nan output

#you can also use numpy to calculate the variance
var_ = np.var(y, ddof=1)
var_
#or
var_ = y.var(ddof=1)
var_
#nan values in the dataset will result in a nan output. You can use nanvar() to ignore nan values.
#its important to specify delta degrees of freedom (ddof)

##Standard Deviation
#The standard deviation is the square root of the variance.

#you can implement the standard deviation in pure Python
std_ = var_ ** .5
std_

#you can also use the statistics module to calculate the standard deviation
std_ = statistics.stdev(x)
std_
#nan values in the dataset will result in a nan output
#you can provide a mean value to the stdev() function to avoid calculating the mean in the function

#you can also use numpy to calculate the standard deviation
std_ = np.std(y, ddof=1)
std_
#dont forget to specify delta degrees of freedom (ddof) as 1!
#nan values in the dataset will result in a nan output. You can use nanstd() to ignore nan values.

#pandas Series objects ignore nan values by default.
std_ = z_with_nan.std()
std_

##Skewness
#Skewness is a measure of the asymmetry in a dataset.
#positive skewness indicates that the tail on the right side of the distribution is longer or fatter than the tail on the left side.
#negative skewness indicates that the tail on the left side of the distribution is longer or fatter than the tail on the right side.

#you can implement skewness in pure Python
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item-mean_)**2 for item in x)/(n-1)
std_ = var_ ** .5
skew_ = sum((item-mean_)**3 for item in x) * n / ((n-1)*(n-2)*std_**3)
skew_
#the skewness is positive, so x has a right-side tail.

#you can also use scipy.stats.skew() to calculate skewness
scipy.stats.skew(y, bias=False)
#the result is the same as the one calculated in pure Python. The bias parameter is set to false to allow for corrections in the calculation of the skewness.
#the optional nan_policy parameter can accept the values 'propagate', 'raise', and 'omit'. 

#pandas Series objects have a skew() method that ignores nan values by default.
z.skew()

##Percentiles
#Percentiles are used to divide a dataset into groups of equal size.
#3 quartiles divide a dataset into 4 groups of equal size.:
#the first quartile (Q1) is the 25th percentile
#the second quartile (Q2) is the 50th percentile
#the third quartile (Q3) is the 75th percentile

#the statistics module has a function to create quantiles.
x = [-5.0, -1.1 , 0.1, 2.0, 8.0, 12.8, 21, 25.8, 41.0]
statistics.quantiles(x, n=2)
statistics.quantiles(x,n=4, method='inclusive')
#in the above example, the n parameter specifies the number of quantiles to create. 
# The method parameter specifies the method used to calculate the quantiles.

#you can use the numpy percentile() function to determine any percentile in a dataset
y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

#the percentile() function accepts several arguments and can also be used to calculate quartiles.
np.percentile(y, [25, 50, 75])
np.median(y)

#if you want to ignore nan values, you can use the nanpercentile() function
y_with_nan = np.insert(y,2,np.nan)
y_with_nan
np.nanpercentile(y_with_nan, [25, 50, 75])

#pandas Series objects have a quantile() method that ignores nan values by default.
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(.05)
z.quantile(.95)
z.quantile([.25, .5, .75])
z_with_nan.quantile([.25, .5, .75])

##Ranges
#range is the difference between the largest and smallest values in a dataset.

#you can use the function np.ptp() to calculate the range of a dataset
np.ptp(y)
np.ptp(z)
np.ptp(z_with_nan)
np.ptp(y_with_nan)

##Interquartile Range
#the interquartile range (IQR) is the difference between the third and first quartiles.

quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

#pandas objects are a bit different. you label with .75 and .25.
quartiles = z.quantile([.25, .75])
quartiles[.75] - quartiles[.25]

##Summary of Descriptive Statistics
#SciPy and Pandas offer functions to quickly calculate descriptive statistics for a dataset.

#SciPy
result = scipy.stats.describe(y, ddof=1, bias=False)
result
#you can omit ddof=1 as it is the default case. 
#you can forsc bias correction in skewness and kurtosis with bias=False.
#with (.) dot notation you can access particular values.
result.mean
#the result is a named tuple with the following attributes:
#nobs: the number of observations
#minmax: the minimum and maximum values
#mean: the mean
#variance: the variance
#skewness: the skewness
#kurtosis: the kurtosis

#Pandas
result = z.describe()
result
#to access a particular value, you can use the index operator.
result['mean']
#the result is a pandas Series object with the following attributes:
#count: the number of observations
#mean: the mean
#std: the standard deviation
#min: the minimum value
#25%: the first quartile
#50%: the second quartile
#75%: the third quartile
#max: the maximum value

##Correlation
#Correlation is a statistical measure that indicates the extent to which two variables are linearly related.
#the different measures of correlation are:
#positive correlation: when two variables increase or decrease together
#negative correlation: when one variable increases as the other decreases
#no correlation: when there is no relationship between the variables

#the two statistics that measure correlation are covariance and the correlation coefficient.

x = list(range(-10,11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_,y_ =  np.array(x), np.array(y)
x__,y__ = pd.Series(x_), pd.Series(y_)

##Covariance
#sample covariance is a measure of the joint variability of two variables.
#the different measures of covariance are:
#positive covariance: when two variables increase or decrease together
#negative covariance: when one variable increases as the other decreases
#no covariance: when there is no relationship between the variables

#the covariance can be calculated in pure Python using the following formula:
n = len(x)
mean_x, mean_y = sum(x)/n, sum(y)/n
cov_xy = sum((x[k]-mean_x)*(y[k]-mean_y) for k in range(n))/(n-1)
cov_xy

#you can also use the numpy cov() function to calculate the covariance.
cov_matrix = np.cov(x_, y_)
cov_matrix
#the result is a 2x2 matrix with the covariance of x and y in the first row and second column.


#you can also use the pandas Series objects to calculate the covariance.
cov_xy = x__.cov(y__)
cov_xy
#or
cov_matrix = y__.cov(x__)
cov_matrix

#the result is the same as the one calculated in pure Python.

##Correlation Coefficient
#the correlation coefficient is a normalized measure of covariance.
#the correlation coefficient is a value between -1 and 1 that indicates 
#the strength and direction of a linear relationship between two variables.

# the correlation coefficient (r) can be interpreted as follows:
#r = 1: perfect positive correlation
#r ≈ 0: no correlation
#r = -1: perfect negative correlation
#r > 0: positive correlation
#r < 0: negative correlation

#the correlation coefficient can be calculated in pure Python using the following formula:
var_x = sum((item - mean_x)**2 for item in x)/(n-1)
var_y = sum((item - mean_y)**2 for item in y)/(n-1)
std_x, std_y = var_x**0.5, var_y**0.5
r = cov_xy/(std_x*std_y)
r

#you can use scipy.stats.pearsonr() to calculate the correlation coefficient. 
#the function returns the correlation coefficient and the p-value.
r, p = scipy.stats.pearsonr(x_, y_)
r
p

#you can also use the numpy corrcoef() function to calculate the correlation coefficient.
#function returns a 2x2 matrix with the correlation coefficient of x and y in the first row and second column.
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

#you can also use linear regression to calculate the correlation coefficient.
result = scipy.stats.linregress(x_,y_)
r = result.rvalue
r

#you can also use the pandas Series objects to calculate the correlation coefficient.
r = x__.corr(y__)
r
#or
r = y__.corr(x__)
r

###Working with 2D Data
##Axes
a = np.array([[1, 1, 1],
             [2, 3, 1],
             [4, 9, 2],
             [8, 27, 4],
             [16, 1, 1]])
#you can apply the same functions to 2D data as you did to 1D data.
np.mean(a)
np.median(a)
np.var(a, ddof=1)
#the functions have an option parameter axis that allows you to specify the axis along which to calculate the statistic.
#axis=none: the default case. the function is applied to all elements in the array.
#axis=0: calculate the statistic across all rows
#axis=1: calculate the statistic across all columns

np.mean(a, axis=0)

#the result is an array with the mean of each column.

np.mean(a, axis=1)

#the result is an array with the mean of each row.

#the axis parameter can be used with other numpy functions.
np.median(a, axis=0)
np.median(a, axis=1)
np.var(a, ddof=1, axis=0)
np.var(a, ddof=1, axis=1)

#SciPy statistics are similar.
scipy.stats.gmean(a) #default axis = 0
scipy.stats.gmean(a, axis=0)
scipy.stats.gmean(a, axis=1)
scipy.stats.gmean(a, axis=None) #statistic is applied to all elements in the array

result = scipy.stats.describe(a, axis =1, ddof=1, bias=False)
result.mean

##DataFrames
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df
#DF methods are similar to series methods. if you call a stats method on a DF, it will be applied to each column by default.
df.mean()
df.var()
#if you want to apply the method to each row, you can use the axis parameter.
df.mean(axis=1)
df.var(axis=1)

#you can isolate a column or row using the column name or row name.
df['A'].mean()
df['A'].var()

df.loc['first'].mean()
df.loc['first'].var()

#sometimes you may want to use an DF as a numpoy array.
#you can use the values attribute to get a numpy array.
df.values
#you can also use the to_numpy() method.
df.to_numpy()

#the .describe() method is also available for DFs.
df.describe()

#the resulting DF has the mean, standard deviation, min, max, and quartiles for each column.

#you can access the values in the resulting DF:

df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

###Visualizing Data

plt.style.use('ggplot')

##Boxplots 
#boxplots are a useful way to visualize the distribution of data.

np.random.seed(seed = 0)
x = np.random.randn(1000)
x_ = pd.Series(np.random.randn(1000))
y = np.random.randn(100)
z = np.random.randn(10)

#you can use the boxplot() function to create a boxplot.
fig, ax = plt.subplots()
ax.boxplot((x,y,z), vert=False, showmeans=True,meanline=True, 
            labels=('x', 'y', 'z'), patch_artist=(True),
            medianprops={'linewidth': 2, 'color': 'purple'},
            meanprops={'linewidth': 2, 'color': 'red'},)
plt.show()

#the parameters of .boxplot() are:
#x: the data to plot
#vert: if True, the boxes are vertical. if False, the boxes are horizontal.
#showmeans: if True, the mean is shown.
#meanline: if True, a line is drawn at the mean.
#labels: the labels for each box.
#patch_artist: if True, the boxes are filled with color.
#medianprops: a dictionary of properties for the median line.
#meanprops: a dictionary of properties for the mean line.

##Histograms
#histograms are a useful way to visualize the distribution of data.
#they are particularly useful when there is a large number of data points.
#frequency is used to correspond to each bin.

#the function np.histogram() returns the frequency and the bin edges.
hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges

#you can use the hist() function to visualize the histogram.
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#the parameters of .hist() are:
#x: the data to plot
#bins: the number of bins or the bin edges.
#cumulative: if True, the histogram is cumulative.
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#cumulative values are the sum of the frequencies of all bins up to and including the current bin.

##Pie Charts
#useful way to visualize relative frequencies.

x,y,z = 128,256,1024

#you can use the pie() function to create a pie chart.
fig, ax = plt.subplots()
ax.pie((x,y,z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

##Bar Charts
x = np.arange(21)
#above is an array of integers from 0 to 20.
y = np.random.randint(21, size=21)
err = np.random.randint(21)

#you can use the bar() function to create a bar chart.
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

##X-Y Plots
#represents the pairs of data from two data sets.
#you can optionally create a line of best fit.

x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

##Heatmaps
#can be used to visually show a matrix.
#the colors represent the values in the matrix.

matrix = np.cov(x,y).round(decimals=2)
fix, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0,1), ticklabels=('x', 'y'))
ax.set_ylim(1.5,-0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

#The yellow color represents the highest value.
#The purple color represents the lowest value.

#You can obtain heat map of correlation coefficients using the corrcoef() method.
matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()
