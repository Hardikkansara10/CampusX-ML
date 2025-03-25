import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Data\\Titanic.csv", usecols=["Age", "Fare", "Survived"])
print(df.head())

x = df.drop(columns="Survived")
y = df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train['Age_imputed'] = x_train['Age']
x_test['Age_imputed'] = x_test['Age']

x_train.loc[x_train['Age_imputed'].isnull(), 'Age_imputed'] = (
    x_train['Age'].dropna().sample(x_train['Age'].isnull().sum(), random_state=42).values
)
x_test.loc[x_test['Age_imputed'].isnull(), 'Age_imputed'] = (
    x_train['Age'].dropna().sample(x_test['Age'].isnull().sum(), random_state=42).values
)

# print(x_train.head(), "\n", x_test.head())

sns.kdeplot(x_train["Age"], label="Original", fill=True)
sns.kdeplot(x_train["Age_imputed"], label="Imputed", fill=True)

plt.legend()
plt.show()

#Same random number for every same output
# sampled_value = x_train['Age'].dropna().sample(1, random_state=int(observation['Fare']))


#Automatically add  Missing indicator
#without indicator
imputer=SimpleImputer(strategy="mean")
x_train_imputed=imputer.fit_transform(x_train)
x_test_imputed=imputer.transform(x_test)

lgr=LogisticRegression()
lgr.fit(x_train_imputed,y_train)
y_pred=lgr.predict(x_test_imputed)
print(accuracy_score(y_test,y_pred))

#with indicator
imputer=SimpleImputer(add_indicator=True)
x_train_imputed=imputer.fit_transform(x_train)
x_test_imputed=imputer.transform(x_test)

lgr.fit(x_train_imputed,y_train)
y_pred=lgr.predict(x_test_imputed)

print(accuracy_score(y_test,y_pred))