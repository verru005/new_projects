#!/usr/bin/env python
# coding: utf-8

# ### HR - Attrition Analytics -  Exploratory Analysis & Predictive Modeling
# > Human Resources are critical resources of any organiazation. Organizations spend huge amount of time and money to hire <br>
# > and nuture their employees. It is a huge loss for companies if employees leave, especially the key resources.  <br>
# > So if HR can predict weather employees are at risk for leaving the company, it will allow them to identify the attrition  <br>
# > risks and help understand and provie necessary support to retain those employees or do preventive hiring to minimize the  <br>
# > impact to the orgranization.

# ### DATA ATRRIBUTES
# 
# satisfaction_level: Employee satisfaction level <br>
# last_evaluation: Last evaluation  <br>
# number_project: Number of projects  <br>
# average_montly_hours: Average monthly hours <br>
# time_spend_company: Time spent at the company <br>
# Work_accident: Whether they have had a work accident <br>
# promotion_last_5years: Whether they have had a promotion in the last 5 years <br>
# department: Department <br>
# salary: Salary <br>
# left: Whether the employee has left <br>

# In[1]:


import pandas as pd
import numpy as np


# In[43]:


# Load the data
hr_df = pd.read_csv( 'HR_comma_sep.csv' )


# In[44]:


hr_df.columns


# In[12]:


hr_df.head()


# In[13]:


hr_df.info()


# In[6]:


#missings
hr_df.isnull().any().sum()


# In[7]:


hr_df.describe().T


# The summary statistics for Work_accident, left and promotion_last_5years does not make sense, as they are categorical variables

# ### EXPLORATORY ANALYSIS

# In[8]:


# 0. How many records of people leaving the company exist in the dataset?
hr_left_df = pd.DataFrame( hr_df.left.value_counts() )
hr_left_df


# In[9]:


#1. What is the percentage of churn by salary bucket


# In[10]:


salary_count = hr_df[['salary', 'left']].groupby(['salary', 'left']).size().reset_index()
salary_count.columns = ['salary', 'left', 'count']


# In[11]:


salary_count


# In[12]:


salary_count = hr_df[['salary', 'left']].groupby(['salary', 'left']).size()
salary_percent = salary_count.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()


# In[13]:


salary_percent


# In[14]:


import matplotlib as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


sn.barplot( hr_left_df.index, hr_left_df.left )


# In[16]:


# 2. How many people, who had work accidents, actually left the company?
work_accident_count = hr_df[['Work_accident', 'left']].groupby(['Work_accident', 'left']).size().reset_index()
work_accident_count.columns = ['Work_accident', 'left', 'count']

sn.factorplot(x="Work_accident", y = 'count', hue="left", data=work_accident_count,
               size=4, kind="bar", palette="muted")


# In[17]:


work_accident_count = hr_df[['Work_accident', 'left']].groupby(['Work_accident', 'left']).size()
work_accident_percent = work_accident_count.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()


# In[18]:


work_accident_percent.columns = ['Work_accident', 'left', 'percent']


# In[19]:


sn.factorplot(x="Work_accident", y = 'percent', hue="left", data=work_accident_percent,
               size=4, kind="bar", palette="muted")


# In[20]:


#3. How work accidents have impacted the statisfactin level of the empolyees?
sn.distplot( hr_df[hr_df.Work_accident == 1]['satisfaction_level'], color = 'r')
sn.distplot( hr_df[hr_df.Work_accident == 0]['satisfaction_level'], color = 'g')


# In[21]:


#4. How satisfaction levels influence whether to stay or leave the company?
sn.distplot( hr_df[hr_df.left == 0]['satisfaction_level'], color = 'g')
sn.distplot( hr_df[hr_df.left == 1]['satisfaction_level'], color = 'r')


# It can be noted, large number of people who had lower satisfaction levels, have left the company. 
# Especially, people who have satisfaction level less than 0.5. This makes sense. But there is also a surge in 
# at higher level of satisfaction. Need to understand and deal with these employees with a different stategy.

# In[22]:


#5. Average satisfaction levels for people who leave and stay back in the company
sl_left_mean = np.mean( hr_df[hr_df.left == 0]['satisfaction_level'] )
sl_left_mean


# In[23]:


np.std( hr_df[hr_df.left == 0]['satisfaction_level'] )


# In[24]:


np.mean( hr_df[hr_df.left == 1]['satisfaction_level'] )


# In[25]:


np.std( hr_df[hr_df.left == 1]['satisfaction_level'] )


# ### Hypothesis Test: Does lower satisfaction levels lead to people leaving the company
# H0 : Average satisfaction level of people leaving is same as average satisfaction of people staying <br>
# H1 : Average satisfaction level of people leaving is less than as average satisfaction of people staying   

# In[26]:


from scipy import stats

stats.ttest_ind( hr_df[hr_df.left == 1]['satisfaction_level'], hr_df[hr_df.left == 0]['satisfaction_level'])


# The test establishes that the average satisfaction levels are different.

# In[27]:


# 6. How last evaluation scores influencing whether to stay or leave the company?
sn.distplot( hr_df[hr_df.left == 0]['last_evaluation'], color = 'r')
sn.distplot( hr_df[hr_df.left == 1]['last_evaluation'], color = 'g')


# People with low evaluation and very high evaluation are leaving, where as people with average evaluation scores are staying back. That seems interesting.

# In[28]:


# 7. How time spent in company influences attrition?
time_spend_count = hr_df[['time_spend_company', 'left']].groupby(['time_spend_company', 'left']).size()
time_spend_percent = time_spend_count.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()
time_spend_percent.columns = ['time_spend_company', 'left', 'percent']


# In[29]:


sn.factorplot(x="time_spend_company", y = 'percent', hue="left", data=time_spend_percent,
               size=4, kind="bar", palette="muted")


# People who have spent 2 years are not leaving the company. But as experience grows people start leaving and highest after they spend 5 years in the company. But once they cross the golden years '7', they are not leaving.

# In[30]:


# 8. Which department has maximum attrition?

dept_count = hr_df[['department', 'left']].groupby(['department', 'left']).size()
dept_count_percent = dept_count.groupby(level=[0]).apply(lambda x: x / x.sum()).reset_index()
dept_count_percent.columns = ['dept', 'left', 'percent']
sn.factorplot(y="dept",
            x = 'percent',
            hue="left",
            data = dept_count_percent,
            size=6,
            kind="bar",
            palette="muted")


# The percentage of people leaving the company is evenly distributed across all depts. Surprisingly, the percentage is high in HR itself. Lowest in management.

# In[31]:


# 9. Effect of whether someone got promoted in last 5 years?
pd.crosstab( hr_df.promotion_last_5years, hr_df.left )


# In[32]:


sn.factorplot(x="promotion_last_5years", hue = 'left', data=hr_df,
               size=4, kind="count", palette="muted")


# Very few people who got promoted in last 5 years left the company, compared to people who are not promoted in last 5 years

# In[33]:


#10.  How Salary is influencing attrition decisions?
sn.factorplot(x="salary", hue = 'left', data=hr_df,
               size=4, kind="count", palette="muted")


# In[34]:


#11. Does higher salary lead to higher satisfaction level?
sn.distplot( hr_df[hr_df.salary == 'low']['satisfaction_level'], color = 'b')
sn.distplot( hr_df[hr_df.salary == 'medium']['satisfaction_level'], color = 'g')
sn.distplot( hr_df[hr_df.salary == 'high']['satisfaction_level'], color = 'r')


# In[35]:


#12.How salaries across departments are related to attrition?
sn.factorplot( y = "department",
            col="salary",
            hue = "left",
            data=hr_df,
            kind="count",
            size=5)


# No surprises. People with lowers salary have maximum percentage of exodus, while people with higher salary the exodus is least.

# In[36]:


# 13. Lets check corrleation between Variables
corrmat = hr_df.corr()
f, ax = plt.pyplot.subplots(figsize=(6, 6))
sn.heatmap(corrmat, vmax=.8, square=True, annot=True)
plt.pyplot.show()


# # Some key observations:
# Satisfaction level reduces as people spend more time in the company. Also, interestingly when they work on more number of projects. <br>
# Evaluation score is positively correlated with spending more montly hours and number of projects. <br>
# As satisfaction level reduces, people tend to leave company <br>

# ### PREDICTIVE MODEL: Build a model to predict if an employee will leave the company

# In[21]:


hr_df.columns


# In[55]:


hr_df.info


# In[4]:


# Encoding Categorical Features
numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project',
     'average_montly_hours', 'time_spend_company']

categorical_features = ['Work_accident','promotion_last_5years', 'department', 'salary']


# In[5]:


# An utility function to create dummy variable
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname)
    col_dummies.drop(col_dummies.columns[0], axis=1, inplace=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return df


# In[6]:


for c_feature in categorical_features:
  hr_df = create_dummies( hr_df, c_feature )


# In[19]:


hr_df.head()


# In[7]:


#Splitting the data

feature_columns = hr_df.columns.difference( ['left'] )
feature_columns1 = feature_columns[1:5]


# In[57]:


feature_columns1


# In[8]:


from sklearn.cross_validation import train_test_split


train_X, test_X, train_y, test_y = train_test_split( hr_df[feature_columns],
                                                  hr_df['left'],
                                                  test_size = 0.2,
                                                  random_state = 42 )


# In[9]:


# Building Models
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit( train_X, train_y )


# In[10]:


list( zip( feature_columns, logreg.coef_[0] ) )


# In[11]:


logreg.intercept_


# In[12]:


#Predicting the test cases
hr_test_pred = pd.DataFrame( { 'actual':  test_y,
                            'predicted': logreg.predict( test_X ) } )


# In[13]:


hr_test_pred = hr_test_pred.reset_index()


# In[14]:


#Comparing the predictions with actual test data
hr_test_pred.sample( n = 10 )


# In[15]:


# Creating a confusion matrix

from sklearn import metrics

cm = metrics.confusion_matrix( hr_test_pred.actual,
                            hr_test_pred.predicted, [1,0] )
cm


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


sn.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[18]:


score = metrics.accuracy_score( hr_test_pred.actual, hr_test_pred.predicted )
round( float(score), 2 )


# Overall test accuracy is 78%. 

# The model is predicting the probability of him leaving the company is only 0.027, which is very low.

# In[37]:


auc_score = metrics.roc_auc_score( hr_test_pred.actual, hr_test_pred.Left_1  )
round( float( auc_score ), 2 )

