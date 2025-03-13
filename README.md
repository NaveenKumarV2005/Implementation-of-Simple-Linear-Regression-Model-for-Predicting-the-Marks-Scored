# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naveen Kumar V
RegisterNumber:  212223220068
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

## Head and Tail value
![Screenshot 2025-03-08 142449](https://github.com/user-attachments/assets/c6d995fc-35ec-4c11-8363-d897261c6d0b)
## Compared Dataset
![Screenshot 2025-03-08 142522](https://github.com/user-attachments/assets/be04fe5e-7779-4c14-8010-f30d1d9a11cf)
## Predicted Values of X nad Y
![Screenshot 2025-03-08 142538](https://github.com/user-attachments/assets/399555ad-475b-40ed-bdf2-655eafb3d236)
## Training sets
![Screenshot 2025-03-08 141806](https://github.com/user-attachments/assets/002dc987-d4ba-45f0-9a7a-41413d84d2ee)
![Screenshot 2025-03-08 141749](https://github.com/user-attachments/assets/5541eb13-54a8-4fba-aeb2-0367f2c42c79)
## MES,MAE,RMSE
![Screenshot 2025-03-08 141813](https://github.com/user-attachments/assets/19cfb52d-7e1c-4317-bc7e-43dcf5ac5109)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
