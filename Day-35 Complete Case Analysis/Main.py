import pandas
import matplotlib.pyplot as plt

df=pandas.read_csv("Data\\Data_science_job.csv")
# print(df.isnull().mean()*100)

caa_cols=[val for val in df.columns if df[val].isnull().mean()*100 <5 and df[val].isnull().mean()*100 >0 ]

new_df=df[caa_cols].dropna()
print(df[caa_cols].sample(5))

# new_df.hist(bins=50,density=True,figsize=(12,12))
# plt.show()

fig=plt.figure()
ax=fig.add_subplot(111)
# df["training_hours"].hist(bins=50,ax=ax,density=True,color="red")
# new_df["training_hours"].hist(bins=50,ax=ax,density=True,color="green",alpha=0.8)

df["city_development_index"].plot.density(color="red")
new_df["city_development_index"].plot.density(color="green")
plt.show()