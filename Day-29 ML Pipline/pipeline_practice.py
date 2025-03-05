import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r'Day-29 ML Pipline\Data.csv')

# Split the data    
X = df.drop('Purchased', axis=1)
y = df['Purchased'] 

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing (Impute + Encode + Scale)
preprocessor = ColumnTransformer([
    ('impute_scale', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), [1, 2]),
    
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), [0])
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the model
pipeline.fit(x_train, y_train)

# Predict and Evaluate
y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
