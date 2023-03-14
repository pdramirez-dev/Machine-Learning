import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
import math
#% read data of csv%
dataset=pd.read_csv("G:\\Proyecto\\IA\\ML\\Eje\\linear_regresion\\data\\house_prices.csv")
size=dataset['sqft_living']
price=dataset['price']


#% handles array data frame%
x=np.array(size).reshape(-1,1)
y=np.array(price).reshape(-1,1)


#% Use Linear regresion%
model=LinearRegression()
model.fit(x,y)



#% MSE and R values%
regre_model_mse=mean_squared_error(x,y)
print("MSE :" ,math.sqrt(regre_model_mse))
print("Valor de R cuadrado:",model.score(x,y))

print(model.coef_[0])
print(model.intercept_[0])

plt.scatter(x,y,edgecolors='green')
plt.plot(x,model.predict(x),color='black')
plt.title("Regresion Lineal")
plt.xlabel("Size")
plt.ylabel("Price")

print("Prediccion :",model.predict([[2000]]))