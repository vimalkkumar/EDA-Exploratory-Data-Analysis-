import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# Importing the dataset 
iris = pd.read_csv('Dataset/Iris.csv')

#### Objective ####
# which features play an important role to finding out the good fit?

"""
Conclusion:
    1. After understanding the data table we can see that the Id column is not playing the important role to approching the OBJECTIVE.
"""
# Triming the Id feature
iris = iris.iloc[:, 1:6]
iris.info
# What are the columns names are available?
iris.columns

# How many Data points and features?
iris.shape

# How many unique values in each features?
iris.nunique(axis = 0)

# Describing the whole dataset like Count, Mean Std, Min, Max 
details = iris.describe()

# how many Null Values are avilable in the dataset?
iris.isnull().sum()

"""
Conclusion:
    We can see that there is no null value is present in the current dataset.
"""

# How many Data-points for each class are presents?
iris['Species'].value_counts()

"""
Conclusion:
    1. Data-points are equally distributed with each class (Iris-virginica - 50, Iris-versicolor - 50 and Iris-setosa - 50). 
    So it is Purely Balanced Dataset.
    2. Note: It is important to check, wether the dataset is BALANCED or IMBALANCED
"""
# Is the different features have any correlations?
features_correlation = iris.corr()
sbn.heatmap(features_correlation, xticklabels = features_correlation.columns, \
            yticklabels = features_correlation.columns, annot = True)
"""
Conclusion:
   PetalWidthCm ia highly correlated with the PetalLengthCm 
"""

# Checking the relationship between the data-points in 3-D plane
sbn.pairplot(iris, hue = 'Species')

"""
Conclusion:
    1. We can see, with the help of 'SepalLengthCm' and 'PetalWidthCm' clearly distanguish the 'Iris-Setosa' 
    flowers and near to distanguish the 'Iris-versicolor' and  'Iris-virginica'.
    2. We have other option too, like with the help of 'PetalLengthCm' and 'PetalWidthCm' we will also distanguish
    the 'Irish-setosa' flowers and may be near to identify the 'Iris-versicolor' and  'Iris-virginica'.
    3. 
"""

# Checking the relationship between the data-points using the 'SepalLengthCm' and 'PetalWidthCm' in 2-D plane
sbn.FacetGrid(iris, hue = 'Species', size = 4) \
    .map(plt.scatter, 'SepalLengthCm', 'PetalWidthCm').add_legend()

"""
Conclusion:
    1. We can se that 'Iris-setosa' is easily seperated
    2. We will write an equation to find each class, like 
        - if pw < 0.7 && sl < 6.0
            'Iris-setosa'
        - if pw >= 1.0 && pw <= 1.7 && sl > 4.8 && sl <= 7.0
            'Iris-versicolor'
        - if pw >= 2.5 && pw > 1.7 && sl > 4.8 && sl <= 8.0
            'Iris-verginica'
    3. We can find simple lines and some if-else statement to build a simple model to classify the flowers types
"""

# Checking the relationship between the data-points using the 'PetalLengthCm' and 'PetalWidthCm' in 2-D plane
sbn.FacetGrid(iris, hue = 'Species', size = 4) \
    .map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm').add_legend()

"""
Conclusion:
    1. We can se that 'Iris-setosa' is easily seperated
    2. We will write an equation to find each class, like 
        - if pw < 0.7 && pl < 2.0
            'Iris-setosa'
        - if pw >= 1.0 && pw <= 1.7 && pl > 2.5 && pl <= 5.0
            'Iris-versicolor'
        - if pw >= 2.5 && pw > 1.7 && pl > 4.5 && pl <= 8.0
            'Iris-verginica'
    3. We can find simple lines and some if-else statement to build a simple model to classify the flowers types
"""

#### Univariate Analysis #####
# Relation with 'SepalLengthCm'
sbn.FacetGrid(iris, hue = 'Species', ) \
    .map(sbn.distplot, 'SepalLengthCm') \
        .add_legend()

"""
Conclusion:
    1. In between 5 to 6 Histrogram, the Distplot shows that 3 classes are 
    overlapping to each other.
"""

# Relation with 'PetalWidthCm'
sbn.FacetGrid(iris, hue = 'Species', ) \
    .map(sbn.distplot, 'PetalWidthCm') \
        .add_legend()

"""
Conclusion:
    1. With the help of 'PetalLengthCm', we can easily distangush the 'Iris-setosa' class.
    2. There are something overlapping with the 'Iris-verginica' and 'Iris-versicolor' classes.
"""

# Relation with 'PetalLengthCm'
sbn.FacetGrid(iris, hue = 'Species', ) \
    .map(sbn.distplot, 'PetalLengthCm') \
        .add_legend()

"""
Conclusion:
    1. With the help of 'PetalLengthCm', we can also distangush the 'Iris-setosa' class.
    2. There are little bit overlapping with the 'Iris-verginica' and 'Iris-versicolor' classes.
"""

# Relation with 'SepalWidthCm'
sbn.FacetGrid(iris, hue = 'Species', ) \
    .map(sbn.distplot, 'SepalWidthCm') \
        .add_legend()

"""
Conclusion:
    1. We can see here that almost all the data points are massively overlap with three different classes.
"""

"""
Final thought:
   1. So, 'PetalLengthCm' >(Betterthan) 'PetalWidthCm' > 'SepalWidthCm' > 'SepalLengthCm'
   2. PetalLengthCm and PetalWidthCm are most useful features to distanguish the varoius types of flowers.
"""
# Is there any Outlier in the dataset?
sbn.boxplot(x = 'PetalLengthCm', y = 'Species', data = iris) 

"""
Conclusion:
    1. A techniques is call Inter-Quartile Range (IQR) is used in Minimum, 25th (First Quartile), 
    50th (Median {Second Quartile}), 75th percentiles (Third Quartile) and Maximum
    2. It is also used for detect the outlier in data set.
    3. These percentiles are also known as the lower quartile, median and upper quartile.
    4. 25% of 'Iris-virginica' is also a 'Iris-versicolor' 
    5. We get 3 outlier is in the PetalLengthCm according to seaborn PetalLengthCm
"""
sbn.boxplot(x = 'PetalWidthCm', y = 'Species', data = iris)

# Where the data-points are denser and sparser?
sbn.violinplot(x = 'PetalLengthCm', y = 'Species', data = iris)

"""
Conclusion:
    1. Denser regions of the data-points are fatter, and sparser ones thinner
"""