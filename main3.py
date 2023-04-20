import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./weather.csv')
print(dataset.shape)
print(dataset[dataset['Sunshine'].isnull()])
dt2=dataset.dropna(subset='Sunshine')

dt2.plot(x='Rainfall',y='Sunshine',style='o')
plt.xlabel('Rainfall')
plt.ylabel('Sunshine')
plt.title('Rain fall vs Sunshine')
plt.show()

plt.figure(figsize=(5,8))
sb.histplot(dt2['Sunshine'],kde=True)
plt.tight_layout()
plt.show()

x=dt2['Rainfall'].values.reshape(-1,1)
y=dt2['Sunshine'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)

print('Intercept: ',regressor.intercept_)
print('Cofficeant: ',regressor.coef_)

y_pred = regressor.predict(x_test)
data1 = pd.DataFrame({'Actual ': y_test.flatten(), 'Predicted ': y_pred.flatten()})
print(data1)

data2 = data1.head(20)
data2.plot(kind='bar' , figsize=(10,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()

plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,y_pred,color='red',linewidth=2)
plt.show()

print('Mean absolute error is: ',metrics.mean_absolute_error(y_test,y_pred))
print('Mean squire error is: ',metrics.mean_squared_error(y_test,y_pred))
print('Root mean squire error is: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))