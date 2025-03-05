import pandas as pd
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OneHotEncoder,  MinMaxScaler

# Load Dataset
df = pd.read_csv('Data\Titanic.csv')
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df.drop(columns=columns_to_drop, inplace=True)
# print(df.head())

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2, random_state=0)

# Imputer
imput_transformer = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough')

# OneHotEncoder
onehot_transformer = ColumnTransformer([
    ('ohe_sex_embarked',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
],remainder='passthrough')

# Scaling
scaler_transformer = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
])

# Feature Selection
# feature_scaler = SelectKBest(score_func=chi2, k=8)

# Model
decision_tree = DecisionTreeClassifier()

# Pipeline
pipe = Pipeline([
    ('imputer', imput_transformer),
    ('onehot', onehot_transformer),
    ('scaler', scaler_transformer),
    # ('feature_selection', feature_scaler),
    ('model', decision_tree)
])


# Fit the Pipeline
pipe.fit(x_train, y_train)
set_config(display='diagram')

#Extracing Info for debug
print(pipe.named_steps['imputer'].transformers_[1][1].statistics_)

y_pred = pipe.predict(x_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))