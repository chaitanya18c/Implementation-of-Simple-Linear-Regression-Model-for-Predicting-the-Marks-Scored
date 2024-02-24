# EX02: Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
**1.** Import the standard Libraries.<br>
**2.** Set variables for assigning dataset values.<br>
**3.** Import linear regression from sklearn.<br>
**4.** Assign the points for representing in the graph.<br>
**5.** Predict the regression for marks by using the representation of the graph.<br>
**6.** Compare the graphs and hence we obtained the linear regression for the given data.<br>
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: CHAITANYA P S  
RegisterNumber: 212222230024
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
y=df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
Regression = LinearRegression()
Regression.fit(X_train,y_train)
y_pred = Regression.predict(X_test)
print(y_pred)
print(y_test)

plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,Regression.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,Regression.predict(X_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

MSE=mean_squared_error(y_test,y_pred)
print("Mean Square Error =",MSE)
MAE=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',MAE)
RMSE=np.sqrt(MSE)
print("Root Mean Square Error =",rmse)
```
## Output:
    
### Head:
![307491335-cf811a1d-04c7-416b-8438-88898d9291b7](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/9bf1bf11-d689-470e-b2d9-f717a217829e)

### Tail:
![307491536-31aa089a-4517-4beb-bcb7-40f594ce0f5f](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/de8ad15b-0b4b-4d3a-9e54-d1513fc3282b)

### X and Y values:
![307491864-fa8422c0-8ee6-4539-a2a6-d3292e9eb5ba](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/49099f22-15f7-4b2a-a604-d58e0c594f0b)

![307491894-ad2b7243-57f6-41d2-be58-72522fc08764](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/3867043f-28d5-4a68-861b-ecb58cc06fbe)

### Values of Y Prediction:
![307492250-c478e878-44b3-46e8-8701-1e756ea00ccd](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/6c9199be-890d-46b5-850a-dd3f577e8d8b)

### Values of Y Test:
![307492268-248564b6-f52c-44d1-9114-285ce185f398](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/5ebb553f-e7b3-4f12-8d04-3c09f3a450e9)

### Training Set:
![image](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/8dbfe716-e304-4848-85eb-b7e6f5cf0bf0)

### Test Set:
![image](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/edbdfe59-435f-487d-9921-c5017da23f71)

### Error Calculation:
![image](https://github.com/chaitanya18c/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119392724/426c3c6e-b3af-45ad-88d7-a7f22d8afcdc)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
