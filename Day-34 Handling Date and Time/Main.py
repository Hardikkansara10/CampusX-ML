import pandas as pd

df=pd.read_csv("Data\\Orders.csv")
# print(df.info())

df['date']=pd.to_datetime(df['date'])
# print(df.info())

df['date_year']=df['date'].dt.year
df['date_month']=df['date'].dt.month
df['date_month_name']=df['date'].dt.month_name()
df['date_day']=df['date'].dt.day
df['date_day_name']=df['date'].dt.day_name()
print(df.head(5))

#TIME
df=pd.read_csv("Data\\Messages.csv")
df=df.rename(columns={"date":"time"})
df=df.drop(columns=["msg"])

df['time']=df['time'].apply(lambda s: s.split()[-1])
df['time']=pd.to_datetime(df['time'])
df['time_hours']=df['time'].dt.hour
df['time_minuts']=df['time'].dt.minute
df['time_seconds']=df['time'].dt.second
print(df.head(5))