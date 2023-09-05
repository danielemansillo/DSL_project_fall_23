#2. "Exploratory Data Analysis(EDA): Python
# Learning the basics of Exploratory Data Analysis using Python
# with Numpy, Matplotlib, and Pandas"
# Code: https://towardsdatascience.com/exploratory-data-analysis-eda-python-87178e35b14

#Other codes:
# https://medium.com/data-science-everywhere/data-preprocessing-a-practical-guide-1b1ce3e884d8
# https://shailchudgar005.medium.com/data-preprocessing-with-scikit-learn-covid-19-dataset-679382532d66
#1. https://medium.com/@ugursavci/complete-exploratory-data-analysis-using-python-9f685d67d1e4

# TABLE OF CONTENTS
### 1. 
### 2.
### 3.
### 4.
### 5.

# BEGIN


# 1. Data Sourcing
#import the useful libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

# Read the data set of "Marketing Analysis" in data.
data= pd.read_csv("marketing_analysis.csv")

# Printing the data
print(data.head())


# 2. Data Cleaning

# Skip the first two rows and load the data again
# Read the file in data without first two rows as it is of no use.
data = pd.read_csv("marketing_analysis.csv",skiprows = 2)
#print the head of the data frame.
print(data.head())

# Drop the customer id as it is of no use.
data.drop('customerid', axis = 1, inplace = True)
#Extract job  & Education in newly from "jobedu" column.
#l = data["jobedu"].split(",")
#data['job'] = l[0]
data['job']= data["jobedu"].apply(lambda x: x.split(",")[0])
data['education']= data["jobedu"].apply(lambda x: x.split(",")[1])
# Drop the "jobedu" column from the dataframe.
data.drop('jobedu', axis = 1, inplace = True)
# Printing the Dataset
print(data.head())

# Checking the missing values
print(data.isnull().sum())
# Dropping the records with age missing in data dataframe.
data = data[~data.age.isnull()].copy()
# Checking the missing values in the dataset.
print(data.isnull().sum())

# Find the mode of month in data
month_mode = data.month.mode()[0]
# Fill the missing values with mode value of month in data.
data.month.fillna(month_mode, inplace = True)
# Let's see the null values in the month column.
print(data.month.isnull().sum())

#drop the records with response missing in data.
# Calculate the missing values in each column of data frame
data = data[~data.response.isnull()].copy()
print(data.isnull().sum())


#Handling Outliers
# Let's calculate the percentage of each job status category.
print(data.job.value_counts(normalize=True))

#plot the bar graph of percentage job categories
plt.figure(figsize=(10,6))
data.job.value_counts(normalize=True).plot.barh()
plt.show()

#calculate the percentage of each education category.
print(data.education.value_counts(normalize=True))

#plot the pie chart of education categories
plt.figure(figsize=(10,6))
data.education.value_counts(normalize=True).plot.pie()
plt.show()

print(data.salary.describe())

#plot the scatter plot of balance and salary variable in data
plt.figure(figsize=(10,6))
plt.scatter(data.salary,data.balance)
plt.show()

#plot the scatter plot of balance and age variable in data
plt.figure(figsize=(10,6))
data.plot.scatter(x="age",y="balance")
plt.show()

#plot the pair plot of salary, balance and age in data dataframe.
sns.pairplot(data = data, vars=['salary','balance','age'])
plt.show()

# Creating a matrix using age, salry, balance as rows and columns
print(data[['age','salary','balance']].corr())

#plot the correlation matrix of salary, balance and age in data dataframe.
plt.figure(figsize=(10,6))
sns.heatmap(data[['age','salary','balance']].corr(), annot=True, cmap = 'Reds')
plt.show()


#groupby the response to find the mean of the salary with response no & yes separately.
print(data.groupby('response')['salary'].mean())

#groupby the response to find the median of the salary with response no & yes separately.
print(data.groupby('response')['salary'].median())

#plot the box plot of salary for yes & no responses.
plt.figure(figsize=(10,6))
sns.boxplot(data.response, data.salary)
plt.show()

#c) Categorical — Categorical Analysis
#create response_rate of numerical data type where response "yes"= 1, "no"= 0
data['response_rate'] = np.where(data.response=='yes',1,0)
print(data.response_rate.value_counts())
#plot the bar graph of marital status with average value of response_rate
plt.figure(figsize=(10,6))
data.groupby('marital')['response_rate'].mean().plot.bar()
plt.show()

#5. Multivariate Analysis
result = pd.pivot_table(data=data, index='education', columns='marital',values='response_rate')
print(result)
#create heat map of education vs marital vs response_rate
plt.figure(figsize=(10,6))
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()