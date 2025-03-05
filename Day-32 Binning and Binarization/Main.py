import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer,Binarizer
from sklearn.model_selection import train_test_split,cross_val_score

"""df=pd.read_csv("Data\\Titanic.csv",usecols=["Age","Fare","Survived"])
df.dropna(inplace=True)
print(df.shape)
x=df.iloc[:,1:]
y=df.iloc[:,:1]
# print(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
print("Before:")
print(accuracy_score(y_test,y_pred))

#Cross Validation
print(cross_val_score(dtc,x,y,cv=7).mean())

#Binning
kbin_age=KBinsDiscretizer(n_bins=15,encode='onehot',strategy='kmeans')
kbin_fare=KBinsDiscretizer(n_bins=15,encode='onehot',strategy='kmeans')

bin_transformer=ColumnTransformer([
    ('bin_age',kbin_age,[0]),
    ('bin_fare',kbin_fare,[1])
])

x_train_bin_transformed=bin_transformer.fit_transform(x_train)
x_test_bin_transformed=bin_transformer.transform(x_test)

# print(bin_transformer.named_transformers_['bin_age'].bin_edges_)

dtc.fit(x_train_bin_transformed,y_train)
y_pred=dtc.predict(x_test_bin_transformed)
print("After")
print(accuracy_score(y_test,y_pred))

print(cross_val_score(dtc,x,y,cv=7).mean())"""


#Binarization

df=pd.read_csv('Data\\Titanic.csv',usecols=['Survived','Age','SibSp','Parch','Fare'])
df.dropna(inplace=True)
# print(df)

df['Family']=df['SibSp']+df['Parch']
df=df.drop(columns=['SibSp','Parch'])
print(df.head())
x=df.drop(columns=['Survived'])
y=df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
print("Before:")
print(accuracy_score(y_test,y_pred))

#Cross Validation
print(cross_val_score(dtc,x,y,cv=7).mean())

binarize_transformer=ColumnTransformer([
    ('binarize_age',Binarizer(copy=False),["Family"])
])

x_train_binarize_transformed=binarize_transformer.fit_transform(x_train)
X_test_binarize_transformed=binarize_transformer.transform(x_test)

dtc.fit(x_train_binarize_transformed,y_train)
y_pred=dtc.predict(X_test_binarize_transformed)
print("After")
print(accuracy_score(y_test,y_pred))
print(cross_val_score(dtc,x,y,cv=7).mean())