import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

df=pd.read_csv('Data\\Titanic_toy.csv')
# print(df.isna().mean())
#Filling missing using Pandas 
df['Age_mean']=df['Age'].fillna(df['Age'].mean())
df['Age_median']=df['Age'].fillna(df['Age'].median())
df['Fare_mean']=df['Fare'].fillna(df['Fare'].mean())
df['Fare_median']=df['Fare'].fillna(df['Fare'].median())
print(df.head(5))

#finding variable diffrance
print('Age',df['Age'].var())
print('Age_mean',df['Age_mean'].var())
print('Age_median',df['Age_median'].var())

print('Fare',df['Fare'].var())
print('fare_mean',df['Fare_mean'].var())
print('fare_median',df['Fare_median'].var())

'''
fig=plt.figure()
# ax=fig.add_subplot(111)
df['Age'].plot.density(color='Red')
df['Age_mean'].plot.density(color='green')
df['Age_median'].plot.density(color='blue')

df['Fare'].plot.density(color='Red')
df['Fare_mean'].plot.density(color='green')
df['Fare_median'].plot.density(color='blue')
plt.show()
'''
X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define transformations
transformer = ColumnTransformer([
    ('age_impute', SimpleImputer(strategy='median'), ['Age']),
    ('fare_impute', SimpleImputer(strategy="mean"), ['Fare'])
], remainder="passthrough")

# Apply transformations
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

columns_transformed = ['Age', 'Fare'] +  X_train.columns.difference(['Age', 'Fare']).tolist()
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=columns_transformed)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=columns_transformed)

print(X_train_transformed_df.head())