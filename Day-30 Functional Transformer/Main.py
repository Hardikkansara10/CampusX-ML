import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer


df = pd.read_csv('Data\Titanic.csv',usecols=['Age','Fare','Survived'])

df.head()

df.isnull().sum()

df['Age'].fillna(df['Age'].mean(),inplace=True)

X = df.iloc[:,1:3]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# plt.figure(figsize=(14,4))
# plt.subplot(121)
# sns.distplot(X_train['Age'])
# plt.title('Age PDF')

# plt.subplot(122)
# stats.probplot(X_train['Age'], dist="norm", plot=plt)
# plt.title('Age QQ Plot')

# plt.show()

# plt.figure(figsize=(14,4))
# plt.subplot(121)
# sns.distplot(X_train['Fare'])
# plt.title('Age PDF')

# plt.subplot(122)
# stats.probplot(X_train['Fare'], dist="norm", plot=plt)
# plt.title('Age QQ Plot')

# plt.show()

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train,y_train)
clf2.fit(X_train,y_train)
    
y_pred = clf.predict(X_test)
y_pred1 = clf2.predict(X_test)
    
print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred1))


trf = FunctionTransformer(func=np.log1p)
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed,y_train)
clf2.fit(X_train_transformed,y_train)
    
y_pred = clf.predict(X_test_transformed)
y_pred1 = clf2.predict(X_test_transformed)
    
print("Accuracy LR",accuracy_score(y_test,y_pred))
print("Accuracy DT",accuracy_score(y_test,y_pred1))