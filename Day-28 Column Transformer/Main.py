import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
df=pd.read_csv(r"Day-28 Column Transformer\covid_toy.csv")
print(df.head())

x_train,x_test,y_train,y_test=train_test_split(df.drop('has_covid',axis=1),df['has_covid'],test_size=0.2)
print(x_train.head())

transformer=ColumnTransformer(transformers=[
    ('se',SimpleImputer(),['fever']),
    ('ms',MinMaxScaler(),['fever']),
    ('oe',OrdinalEncoder(categories=[['Strong','Mild']]),['cough']),
    ('ohe',OneHotEncoder(drop='first',sparse_output=False,dtype=np.int32),['gender','city']),
],remainder='passthrough')

x_train_transformed=transformer.fit_transform(x_train)
x_test_transformed=transformer.transform(x_test)
print(x_train_transformed.sample())
