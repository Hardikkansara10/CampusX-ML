import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data\\Titanic.csv", usecols=['Age', 'Pclass', 'SibSp', 'Parch', 'Survived'])
df.dropna(inplace=True)

x = df.drop(columns="Survived")
y = df["Survived"]

cross_val = cross_val_score(LogisticRegression(), x, y, scoring="accuracy", cv=20).mean()
print("Initial Accuracy:", cross_val)

x['Family_size'] = x['SibSp'] + x['Parch'] + 1

def fun(size):
    if size == 1:
        return 0
    elif size <= 4:
        return 1
    else:
        return 2

x['Family_type'] = x['Family_size'].apply(fun)
x.drop(columns=["SibSp", "Parch", "Family_size"], inplace=True)

cross_val = cross_val_score(LogisticRegression(), x, y, scoring="accuracy", cv=20).mean()
print("Accuracy after Feature Engineering:", cross_val)


df = pd.read_csv("data\\Titanic.csv", usecols=['Name', 'Survived'])
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

print(df[['Title', 'Name']].head())
# print(df.groupby('Title')['Survived'].mean().sort_values(ascending=False))
print(df.groupby('Title')['Survived'].mean() )
