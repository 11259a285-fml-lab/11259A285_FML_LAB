from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
y = iris.data[:,2]
X = iris.data[:,[0,1,3]]

print("Features")
print("features_names")
print("Dataset shape",X.shape)
print("Target shape",y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training samples:",len(X_train))
print('Test samples:',len(X_test))
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print ("Actual vs predicted")
for actual ,predicted in zip(y_test[:10],y_pred[:10]):
    print("actual:",round(actual,2),"predicted:",round(predicted,2))
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("Model Evaluation")
print("mean_absolute_error:",round(mae,2))
print("mean_squared_error:",round(mse,2))
print('root_mean_squared_error:',round(rmse,2))
print("R2:",round(r2,2))
print("models coefficient")
for name,coefficient in zip(["Sepal length","Sepal width","petal width"],model.coef_):
        print("name"":",round(coefficient,2))
        print("coefficient:",round(coefficient,2))







