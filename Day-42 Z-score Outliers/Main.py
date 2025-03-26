import pandas 
import numpy
import seaborn
import matplotlib.pyplot as plt
df=pandas.read_csv("data\\Placement.csv")
print(df.head())

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
seaborn.displot(df["cgpa"])

plt.subplot(1,2,2)
seaborn.displot(df["placement_exam_marks"])
# plt.show()

print("Mean",df["cgpa"].mean())
print("Standered Deviation",df["cgpa"].std())
print("Minimum",df["cgpa"].min())
print("Maximun",df["cgpa"].max())

print("Lowest allowed:", df["cgpa"].mean()-(3*df["cgpa"].std()))
print("Highest allowed:", df["cgpa"].mean()+(3*df["cgpa"].std()))

outlier_df=df[(df["cgpa"]<5.1) | (df["cgpa"]>8.8)]
# print(outlier_df)
 
#TRIMMING Outliers
new_df=df[(df["cgpa"]>5.1) & (df["cgpa"]<8.8)]
# print(new_df.head(5))


#Z SCORE_ METHOD
df["cgpa_zcore"]=(df["cgpa"]-df["cgpa"].mean())/df["cgpa"].std() 
# print(df.head())

new_df=df[(df["cgpa_zcore"]<3)|(df["cgpa_zcore"]>-3)]
print(new_df.head(5))

lower=df["cgpa"].mean()-(3*df["cgpa"].std())
upper=df["cgpa"].mean()+(3*df["cgpa"].std())

df["cgpa"]=numpy.where(
    df["cgpa"]>upper,
    upper,
    numpy.where(
        df["cgpa"]<lower,
        lower,
        df["cgpa"]
    )
)

print(df)