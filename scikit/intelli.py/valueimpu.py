import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer


df = fetch_openml('titanic', version=1, as_frame=True)['data']
# df.info()
# df.isnull().sum()
mvp = pd.DataFrame(df.isnull().sum()/len(df)*100)
mvp.plot(kind='bar',title='missing values',ylabel='values')
# print(f'size of missing value:{df.shape}')

# print(f'no.of vull values before impute:{df.F.isnull().sum()}')

imp = SimpleImputer(strategy='mean')
df['age'] = imp.fit_transform(df[['age']])
# print(f'after impute:{df.age.isnull().sum()}')

def get_parameters(df):
    parameters={}
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype=='float64' or df[col].dtype=='int64' or df[col].dtype=='int32':
            strategy='mean'
        else:
            strategy='most_frequent'
        mv = df[col][df[col].isnull()].values[0]
        parameters[col] = {'missing value': mv,'strategy':strategy}
        return parameters
get_parameters(df)

for col, param in get_parameters(df).items():
    mv = param['missing value']
    strategy = param['strategy']
    imp = SimpleImputer(missing_values= mv, strategy=strategy)
    df[col]=imp.fit_transform(df[[col]])

#data encoder
from sklearn.preprocessing import OneHotEncoder
df[['female','male']]=OneHotEncoder().fit_transform(df[['sex']]).toarray()
print(df[['sex','female','male']])

#data scaling 

from sklearn.preprocessing import  StandardScaler, MinMaxScaler
ss = StandardScaler()
nc = df.select_dtypes(include=['int64', 'float64', 'int32']).columns
# print(nc)
# df[nc]=ss.fit_transform(df[nc])
# print(df[nc].describe())
mm = MinMaxScaler()
df[nc] = mm.fit_transform(df[nc])
print(df[nc])