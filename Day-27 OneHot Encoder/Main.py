import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'CampusX ML\Day-27 OneHot Encoder\cars.csv')

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:4], df.iloc[:, -1], test_size=0.2, random_state=42)

ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)
x_train_ohe = ohe.fit_transform(x_train[['fuel', 'owner']])
x_test_ohe = ohe.transform(x_test[['fuel', 'owner']])

y = np.hstack((x_train[['brand', 'km_driven']].values, x_train_ohe))
counts = df['brand'].value_counts()
threshold = 100
repl = counts[counts <= threshold].index
a = pd.get_dummies(df['brand'].replace(repl, 'other'))

fuel_owner_encoded = ohe.fit_transform(df[['fuel', 'owner']])

y = np.hstack((a, df[['km_driven']].values, fuel_owner_encoded))
print(y)
print(y.shape)
