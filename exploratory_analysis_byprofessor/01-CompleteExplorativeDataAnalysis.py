#1. "Complete Exploratory Data Analysis using Python"
# Code: https://medium.com/@ugursavci/complete-exploratory-data-analysis-using-python-9f685d67d1e4
# Data: https://www.kaggle.com/mirichoi0218/insurance

# import libraries
import numpy as np # linear algebra
import pandas as pd # data manipulation and analysis
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import warnings # ignore warnings

# TABLE OF CONTENTS
### 1. Understanding Business Problem
### 2. Variable Description
### 3. Data Understanding
### 4. Data Cleaning
### 5. Data Visualization

# BEGIN
### 1. Understanding Business Problem
# ....

### 2. Variable Description
# ....

### 3. Data Understanding

sns.set_style('whitegrid') # set style for visualization
warnings.filterwarnings('ignore')

# load the dataset
df = pd.read_csv('insurance.csv')

# get some rows
print(df.head()) # first rows (5 rows as default value)
print(df.sample(5)) # randomly selected rows
print(df.tail()) # last rows

# get s0me information about the dataframe
print(df.info())
print(df.shape) # get number of rows and columns
print(df.columns) # get the attibute names
print(df.describe) # get some descriptive statistics
print(df.describe(include='O')) #????
print(list(df.sex.unique())) # get unique values of an attribute

### 4. Data Cleaning (missing and duplicated values removal)
print(df.isnull().sum()) # ???? summing up the missing values ?????
print(df[df.duplicated(keep='first')]) #????
df.drop_duplicates(keep='first',inplace=True) #???

### 5. Data Visualization (Univariate, Bivariate and Multivariate Analysis)
# We can perform univariate analysis with 3 options :
#     Summary Statistics
#     Frequency Distributions Table
#     Charts ( Boxplot, Histogram, Barplot, Pie Chart)

# 5.1 Univariate Analysis (for numerical and categorical features)
# 5.1.a Univariate Analysis for Numerical Features
# "Charges" feature
plt.figure(figsize=(10,6))
sns.distplot(df.charges,color='r')
plt.title('Charges Distribution',size=18)
plt.xlabel('Charges',size=14)
plt.ylabel('Density',size=14)
plt.show()
# "Age" feature
plt.figure(figsize=(10,6))
sns.histplot(df.age)
plt.title('Age Distribution',size=18)
plt.xlabel('Age',size=14)
plt.ylabel('Count',size=14)
plt.show()
# "BMI" feature
plt.figure(figsize=(10,6))
plt.hist(df.bmi,color='y')
plt.title('BMI Distribution',size=18)
plt.show()
# BoxPlot for numerical features
plt.figure(figsize = (10,6))
sns.boxplot(df.charges)
plt.title('Distribution Charges',size=18)
plt.show()
# compute Q1, Q3 and IQR
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
#select outliers
outliers = df[(df['charges']< Q1-1.5* IQR) | (df['charges']> Q3+1.5* IQR)]
#remove outliers and get the new dataset (without outliers)
df = df[(df['charges']>= Q1-1.5* IQR) & (df['charges'] <= Q3+1.5* IQR)]
#or df = df[~((df['charges']< Q1-1.5* IQR) | (df['charges']> Q3+1.5* IQR))]

# BoxPlot of "charges" after outliers removal
plt.figure(figsize = (10,6))
sns.boxplot(df.charges)
plt.title('Distribution Charges',size=18)
plt.show()

# 5.1.b Univariate Analysis for Categorical Features
# "Gender" feature
plt.figure(figsize=(10,6))
sns.countplot(x = 'sex', data = df)
plt.title('Total Number of Male and Female',size=18)
plt.xlabel('Sex',size=14)
plt.show()
# "Children" feature
plt.figure(figsize = (10,6))
sns.countplot(df.children)
plt.title('Children Distribution',size=18)
plt.xlabel('Children',size=14)
plt.ylabel('Count',size=14)
plt.show()
# "Smoker" feature
plt.figure(figsize = (10,6))
sns.countplot(df.smoker)
plt.title('Smoker Distribution',size=18)
plt.xlabel('Smoker',size=14)
plt.ylabel('Count',size=14)
plt.show()
print(df.smoker.value_counts())
# "Region" feature
plt.figure(figsize = (10,6))
sns.countplot(df.region,palette='Blues')
plt.title('Region Distribution',size=18)
plt.xlabel('Region',size=14)
plt.ylabel('Count',size=14)
plt.show()

# 5.2 Bivariate Analysis (find relationship between two variables)
# For bivariate analysis, we usually use
# - scatterplot (numerical vs numerical)
# - boxplot (categorical vs numerical)
# - contingency table (categorical vs categorical)

# Ages vs Charges (numeric vs numeric)
plt.figure(figsize = (10,6))
sns.scatterplot(x='age',y='charges',color='r',data=df)
plt.title('Age vs Charges',size=18)
plt.xlabel('Age',size=14)
plt.ylabel('Charges',size=14)
plt.show()
# we compute the correlation between the two variables
print('Correlation between age and charges is : '
      '{}'.format(round(df.corr()['age']['charges'],3)))

# Smoker vs charges (categoric vs numeric)
plt.figure(figsize = (10,6))
sns.set_style('darkgrid')
sns.boxplot(x='smoker',y='charges',data=df)
plt.title('Smoker vs Charges',size=18);
plt.show()
# Contingency table
# ????????????????????


#Using Pairplot for Numerical Values

#Pair plot is another awesome method that shows us
# the relationship between two numerical values
# as well as the distribution of each variable.
sns.pairplot(df, markers="+",
                 diag_kind="kde",
                 kind='reg',
                 plot_kws={'line_kws':{'color':'#aec6cf'},
                           'scatter_kws': {'alpha': 0.7,
                                           'color': 'red'}},
                 corner=True);
plt.title('Pairplot',size=18);
plt.show()


# 5.3 Multivariate Analysis (find relationship between two variables)
#There are different methods to calculate correlation coefficient ;
# - Pearson
# - Kendall
# - Spearman
# We will combine the .corr() method with heatmap so that
# we will be able to see the relationship in the graph.
# .corr() method is used Pearson correlation by default.
plt.figure(figsize = (10,6))
sns.heatmap(df.corr(),annot=True,square=True,
            cmap='RdBu',
            vmax=1,
            vmin=-1)
plt.title('Correlations Between Variables',size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()

