import pandas 
import numpy
import seaborn
import matplotlib.pyplot as plt
df=pandas.read_csv("data\\Placement.csv")
# print(df.head())

# plt.figure(figsize=(16,5))
# plt.subplot(1,2,1)
# seaborn.displot(df["cgpa"])

# plt.subplot(1,2,2)
# seaborn.displot(df["placement_exam_marks"])
# # plt.show()

# seaborn.boxplot(df["placement_exam_marks"])
# plt.show()

quantile_1=df["placement_exam_marks"].quantile(0.25)
quantile_3=df["placement_exam_marks"].quantile(0.75)

IQR=quantile_3-quantile_1

Upper=quantile_3+1.5*IQR
Lower=quantile_1-1.5*IQR

print("Upper outliers:",Upper)
print("Lowerr outliers:",Lower)


#TRIMMING
outlier_df=df[(df["placement_exam_marks"]>Upper)|(df["placement_exam_marks"]<Lower)]
new_df=df[(df["placement_exam_marks"]<Upper)&(df["placement_exam_marks"]>Lower)]
# print(new_df.head())


plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
seaborn.histplot(df["placement_exam_marks"], kde=True) 

plt.subplot(2, 2, 2)
seaborn.boxplot(x=df["placement_exam_marks"])

plt.subplot(2, 2, 3)
seaborn.histplot(new_df["placement_exam_marks"], kde=True)

plt.subplot(2, 2, 4)
seaborn.boxplot(x=new_df["placement_exam_marks"])

plt.show()

#CAPPING
new_df=df.copy()
new_df["placement_exam_marks"]=numpy.where(
    new_df["placement_exam_marks"]>Upper,
    Upper,
    numpy.where(
        new_df["placement_exam_marks"]<Lower,
        Lower,
        new_df["placement_exam_marks"]
    )
)
# print(df)


plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
seaborn.histplot(df["placement_exam_marks"], kde=True) 

plt.subplot(2, 2, 2)
seaborn.boxplot(x=df["placement_exam_marks"])

plt.subplot(2, 2, 3)
seaborn.histplot(new_df["placement_exam_marks"], kde=True)

plt.subplot(2, 2, 4)
seaborn.boxplot(x=new_df["placement_exam_marks"])

plt.show()