import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data\\House.csv", usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])
print(df.head())

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


#Using Mode
imputer = SimpleImputer(strategy="most_frequent")
X_train_transformed = imputer.fit_transform(X_train)
X_test_transformed = imputer.transform(X_test)

X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=X_train.columns)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

X_train["GarageQual"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[0], color='red', alpha=0.5, label="Original")
X_train_transformed_df["GarageQual"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[0], color='green', alpha=0.5, label="Transformed")
axes[0].set_title("GarageQual Distribution")
axes[0].legend()

X_train["FireplaceQu"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[1], color='red', alpha=0.5, label="Original")
X_train_transformed_df["FireplaceQu"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[1], color='green', alpha=0.5, label="Transformed")
axes[1].set_title("FireplaceQu Distribution")
axes[1].legend()

plt.tight_layout()
plt.show()

print(X_train_transformed_df.head())


#Creating new column
imputer = SimpleImputer(strategy="constant",fill_value="Missing")
X_train_transformed = imputer.fit_transform(X_train)
X_test_transformed = imputer.transform(X_test)

X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=X_train.columns)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

X_train["GarageQual"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[0], color='red', alpha=0.5, label="Original")
X_train_transformed_df["GarageQual"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[0], color='green', alpha=0.5, label="Transformed")
axes[0].set_title("GarageQual Distribution")
axes[0].legend()

X_train["FireplaceQu"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[1], color='red', alpha=0.5, label="Original")
X_train_transformed_df["FireplaceQu"].astype(str).value_counts(normalize=True).plot(kind="bar", ax=axes[1], color='green', alpha=0.5, label="Transformed")
axes[1].set_title("FireplaceQu Distribution")
axes[1].legend()

plt.tight_layout()
plt.show()

print(X_train_transformed_df.head())