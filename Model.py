import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_csv('DataSet/Salary_dataset.csv')
x=df[['YearsExperience']].values.reshape(-1,1)
y=df[['Salary']].values.reshape(-1,1)
lin=LinearRegression
lin.fit(x,y)
pickle.dump(lin,open('model.pkl','wb'))

