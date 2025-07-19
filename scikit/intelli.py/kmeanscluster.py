import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("scikit/car.data", encoding='unicode_escape')
print(df)
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.dropna())

# df["C"] = pd.to_numeric(df["C"],errors='coerce')
# df["B"] = df["C"] * df["D"]
# df_m = df.groupby("C")["B"].sum()
# df_m = df_m.reset_index()
# print(df_m.head())

att = ['A','B','C']
plt.rcParams['figure.figsize']=[8,8]
sns.boxplot(data=df[att],orient="v", palette="Set2", whis=1.5, saturation=1,width=0.7)
plt.title("outliers")
plt.ylabel("Range", fontweight="bold")
plt.xlabel("Attributes", fontweight='bold')
# plt.show()

df1 = df_retail[['Amount','Frequency', 'Recenct']]
scaler = StandardScaler()
df_retail_scaled = scaler.fit_transform(df1)
df_retail_scaled.shape