import pandas
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV

df=pandas.read_csv("Data\\Titanic.csv")
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
print(df.head())

x=df.drop(columns="Survived")
y=df["Survived"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

numeric_feature_scaling_column=["Age","Fare"]
numeric_feature_scaling=Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

category_feature_scaling_column=["Embarked","Sex"]
category_feature_scaling=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

preprocessing=ColumnTransformer(
    transformers=[
        ('numeric_transformer', numeric_feature_scaling,numeric_feature_scaling_column),
        ('category_transformer', category_feature_scaling,category_feature_scaling_column)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('classifier', LogisticRegression())
])

set_config(display='diagram')
# print(clf)

param_grid = {
    'preprocessor__numeric_transformer__imputer__strategy': ['mean', 'median'],
    'preprocessor__category_transformer__imputer__strategy': ['most_frequent', 'constant'],
    'preprocessor__category_transformer__imputer__fill_value': ['missing'],
    'classifier__C': [0.1, 1.0, 10, 100]
}

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(x_train, y_train)

print(f"Best params:")
print(grid_search.best_params_)

import pandas as pd

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)

print(cv_results[['param_classifier__C',
            'param_preprocessor__category_transformer__imputer__strategy',
            'param_preprocessor__numeric_transformer__imputer__strategy',
            'mean_test_score']])
