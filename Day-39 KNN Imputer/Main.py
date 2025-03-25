import pandas
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score


df=pandas.read_csv("Data\\Titanic.csv",usecols=["Age","Pclass","Fare","Survived"])
# print(df.head())

x = df.drop(columns="Survived")
y = df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#simple imputer
simple_imputer=SimpleImputer(strategy="mean")
x_train_imputed=simple_imputer.fit_transform(x_train)
x_test_imputed=simple_imputer.transform(x_test)

lgr=LogisticRegression()
lgr.fit(x_train_imputed,y_train)
y_pred=lgr.predict(x_test_imputed)
print("Simple imputer:",accuracy_score(y_test,y_pred))
print("Cross Val Score:",cross_val_score(lgr,x_test_imputed,y_test,cv=5).mean())


#KNN imputer
knn_imputer=KNNImputer(n_neighbors=2,weights="distance")
x_train_imputed=knn_imputer.fit_transform(x_train)
x_test_imputed=knn_imputer.transform(x_test)

lgr=LogisticRegression()
lgr.fit(x_train_imputed,y_train)
y_pred=lgr.predict(x_test_imputed)
print("KNN Imputer:",accuracy_score(y_test,y_pred))
print("Cross Val Score:",cross_val_score(lgr,x_test_imputed,y_test,cv=5).mean())