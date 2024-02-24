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
    
### df.head():
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/cf811a1d-04c7-416b-8438-88898d9291b7" width=10%>

### df.tail():
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/31aa089a-4517-4beb-bcb7-40f594ce0f5f" width=10%>

### X and Y values:
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/fa8422c0-8ee6-4539-a2a6-d3292e9eb5ba" width="15%" height="2%">
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/ad2b7243-57f6-41d2-be58-72522fc08764">

### Values of Y Prediction:
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/c478e878-44b3-46e8-8701-1e756ea00ccd">

### Values of Y Test:
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/248564b6-f52c-44d1-9114-285ce185f398">

### Training Set:
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/a781aafb-8ef4-4911-8ad9-77ae5fd4a4a3" width=25%>

### Test Set:
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/626473d3-1c5a-40f8-a3c9-15749751129a" width=25%>

### Error Calculation:
<img src="https://github.com/Janarthanan2/ML_Ex02_Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393515/bb1cba4b-1442-445f-8491-f097661f0813" width=25%>

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
