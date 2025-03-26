import pandas as pd
import numpy as np

df=pd.read_csv('Data\\Titanic_mixed.csv')
# print(df.head())
df=df.rename(columns={'number':'Number'})

df['Number_numeric']=pd.to_numeric(df['Number'],errors='coerce',downcast='integer')
df['Number_category']=np.where(df['Number_numeric'].isnull(),df['Number'],np.nan)
df['Number_numeric']=np.where(df["Number_numeric"].isnull(),0.0,df['Number_numeric'])
df=df.drop(columns=['Number'])

df['Cabin_numeric']=df['Cabin'].str.extract('(\\d+)')
df['Cabni_category']=df['Cabin'].str[0]
df=df.drop(columns=['Cabin'])

df['Ticket_numeric'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['Ticket_numeric'] = pd.to_numeric(df['Ticket_numeric'],errors='coerce',downcast='integer')

df['Ticket_category'] = df['Ticket'].apply(lambda s: s.split()[0])
df['Ticket_category'] = np.where(df['Ticket_category'].str.isdigit(), np.nan,df['Ticket_category'])
df=df.drop(columns=['Ticket'])

print(df.head(10))