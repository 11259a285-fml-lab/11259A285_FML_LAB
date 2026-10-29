from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import(

accuracy_score,
confusion_matrix,
precision_score,
recall_score,
f1_score,
)
iris = load_iris()
X = iris.data
Y = iris.target
print("Dataset Shape:",len(X))
print("Number of samples:",X.shape[0])
print("Number of features:",X.shape[1])
print("features")
print("iris.feature_names")
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
print("Dataset splitting")
print("-----------------")
print("Training samples",len(X_train))
print("Test samples",len(X_test))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = SVC(
    kernel = 'linear',
    C = 1.0
)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    y_pred

)
cm = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test,y_pred,average = 'weighted')
recall = recall_score(y_test,y_pred,average = 'weighted')
f1 = f1_score(y_test,y_pred,average = 'weighted')
print("SVC model evaluation")
print("Accuracy:",round(accuracy,4))
print("Precision:",round(precision,4))
print("Recall:",round(recall,4))
print("F1 score:",round(f1,4))
print("confusion matrix:")
print(cm)

