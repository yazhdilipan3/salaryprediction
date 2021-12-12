import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
list = ["years","salary"]
x = pd.read_csv("salary.csv",usecols = list,names = ["years","salary"])
print(x.head())
print(x.tail())
print(x.describe())
print(x.isnull())
print(x["years"].describe())
plt.plot(x)
plt.title("salarydistribution",fontsize = 45)
plt.xlabel("years",fontsize = 25)
plt.ylabel("salary",fontsize = 25)
plt.show()
from sklearn.model_selection import train_test_split

v = x
x = v.drop("salary",axis = 1)
y = v["salary"]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.2,random_state = None)
from sklearn.linear_model import LinearRegression

L = LinearRegression()
L.fit(x_train,y_train)
y_pred = L.predict(x_test)
print(y_pred)
print(L.score(x_test,y_test))





