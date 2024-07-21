import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = load_diabetes()
print(df)
dataset = pd.DataFrame(df.data)
# print(dataset)-
dataset.columns = df.feature_names
print(dataset.head())
dataset['diabetes'] = df.target   #--> creating new feature called diabetes and it is dependent feature
print(dataset.head())
X = dataset.iloc[:, :-1]   ##independent feature
Y=dataset.iloc[:, -1]    ##dependent feature
print(X.head())
print(Y.head())
linereg=LinearRegression()

###below is for cross validation using k fold and the value is 5
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
mse=cross_val_score(linereg,X_train,Y_train,scoring="neg_mean_squared_error",cv=5)
print(mse)
means_mse=np.mean(mse)
print(means_mse)

# below line to tarin the data x and y for new data
linereg.fit(X_train,Y_train)
predictions=linereg.predict(X_test)    ##this line used to predict for y
print("predictions are:",predictions)
r2=r2_score(predictions,Y_test)
print(r2)









