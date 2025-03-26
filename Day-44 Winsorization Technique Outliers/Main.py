import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

df = pandas.read_csv("Data\\Weight-Height.csv")

# plt.figure(figsize=(16, 10))

# plt.subplot(2, 2, 1)
# seaborn.histplot(df["Height"], kde=True)

# plt.subplot(2, 2, 2)
# seaborn.histplot(df["Weight"], kde=True)

# plt.subplot(2, 2, 3)
# seaborn.boxplot(x=df["Height"])

# plt.subplot(2, 2, 4)
# seaborn.boxplot(x=df["Weight"]) 
# plt.show()


Upper=df["Height"].quantile(.99)
Lower=df["Height"].quantile(.01)
print("Upper outliers:",Upper)
print("Lowerr outliers:",Lower)


#TRIMMING
outlier_df=df[(df["Height"]>=Upper)|(df["Height"]<=Lower)]
print(outlier_df.shape)

new_df=df[(df["Height"]<=Upper)&(df["Height"]>=Lower)]
print(new_df.shape)

# plt.figure(figsize=(16, 10))

# plt.subplot(2, 2, 1)
# seaborn.histplot(new_df["Height"], kde=True)

# plt.subplot(2, 2, 2)
# seaborn.histplot(new_df["Weight"], kde=True)

# plt.subplot(2, 2, 3)
# seaborn.boxplot(x=new_df["Height"])

# plt.subplot(2, 2, 4)
# seaborn.boxplot(x=new_df["Weight"]) 
# plt.show()

#CAPPING
new_df=df.copy()
new_df["Height"]=numpy.where(
    new_df["Height"]>Upper,
    Upper,
    numpy.where(
        new_df["Height"]<Lower,
        Lower,
        new_df["Height"]
    )
)
print(new_df)