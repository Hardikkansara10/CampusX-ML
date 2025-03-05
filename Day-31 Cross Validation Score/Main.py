import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('Data/Purchased.csv')
x=df[["Age","EstimatedSalary"]]
y=df["Purchased"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=5)
knc=KNeighborsClassifier(n_neighbors=5)
knc.fit(x_train,y_train)
y_pred=knc.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))

#cross_validation
print(cross_val_score(knc,x,y,cv=5).mean())