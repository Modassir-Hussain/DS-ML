# %%
import numpy as np # Liner algebra library
import pandas as pd # data manipulation tool
import matplotlib.pyplot as plt#
import seaborn as sns

# %%
# df = pd.read_csv('sample_data/california_housing_train.csv')
df = pd.read_csv('http://bit.ly/kaggletrain')

# %%
df.head()

# %%
df.isnull().sum()

# %%
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')

# %%
"""
## Data Cleaning
"""

# %%
# imputing missing values in 'Age' column according to class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

# %%
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)

# %%
# 'Age' is Imputed 'Age' average value according to 'Pclass'
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')

# %%
# Droping 'Cabin' as it has lots of NaN Value
df.drop('Cabin',axis=1,inplace=True)

# %%
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')

# %%
"""
## Handling categorical columns
"""

# %%
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)

# %%
df.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

# %%
df = pd.concat([df,sex,embark],axis=1)

# %%
df.head()

# %%
"""
## Machien learning
"""

# %%
# Spliting the data
X=df.iloc[:,1:].values
y=df.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# %%
X_test

# %%
from sklearn.linear_model import LogisticRegression

# %%
lm = LogisticRegression(verbose=1)

# %%
lm.fit(X_train,y_train)

# %%
pred = lm.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix
# %%
cr = classification_report(y_test,pred)

# %%
print(cr)

# %%
cm = confusion_matrix(y_test,pred)

# %%
print(cm)

# %%
# since we are creting multiple linear regression model, here i am using the backward elimination technique to evaluate and build the model.
# Backward Elimination consists of the following steps:
# Select a significance level to stay in the model (eg. SL = 0.05)
# Fit the model with all possible predictors
# Consider the predictor with the highest P-value. If P>SL, go to point d.
# Remove the predictor
# Fit the model without this variable and repeat the step c until the condition becomes false.

import statsmodels.api as sm
# add a column of ones as integer data type
x = np.append(arr = np.ones((891, 1)).astype(int),
			values = X, axis = 1)
# choose a Significance level usually 0.05, if p>0.05
# for the highest values parameter, remove that value
x_opt = x[:, [0, 1, 2, 3, 4, 5,6,7,8]]
ols = sm.OLS(endog = y, exog = x_opt).fit()
ols.summary()
#Note
# R-squared:0.400
# Adj.R-squared:0.394
# Skew:0.540
# X7 has the highest value for the parameter 'P>|t|' so removing the X7 i.e 'Q'

# %%
# removing 'Q' and creating new DataFrame df_R

df_R = df.drop(['Q'],axis=1)
df_R.head()
# df.drop(['male'],axis=1,inplace=True)
# df.head()

# %%
# Spliting the data

X=df_R.iloc[:,1:].values
y=df_R.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
# scaling the data

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# %%
# Applying Logistic Regression
from sklearn.linear_model import LogisticRegression

# %%
lm = LogisticRegression(verbose=1)

# %%
lm.fit(X_train,y_train)

# %%
pred = lm.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix

# %%
cr = classification_report(y_test,pred)

# %%
print(cr)

# %%
cm = confusion_matrix(y_test,pred)

# %%
print(cm)

# %%
import statsmodels.api as sm
# add a column of ones as integer data type
x = np.append(arr = np.ones((891, 1)).astype(int), values = X, axis = 1)
# choose a Significance level usually 0.05, if p>0.05
# for the highest values parameter, remove that value
x_opt = x[:, [0, 1, 2, 3, 4, 5,6,7]]
ols = sm.OLS(endog = y, exog = x_opt).fit()
ols.summary()

# After removing 'Q'
# 'R-squared:0.399' is decreased
# Adj.R-squared:0.395 is increased
# Skew:0.541 is increased

# %%
"""
# Conclusion
## Our first modle is performing better than our second model so we will go ahead with our first model
"""