from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
accuracy_score,
confusion_matrix,
classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset splitting")
print("---------------")
print("training samples:",len(X_train))
print("test samples:",len(X_test))
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred,target_names=iris.target_names))
print("Random forest")
print("------------------------")
print("Accuracy",accuracy)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
xticklabels = iris.feature_names
yticklabels = iris.target_names
plt.title("Random forest")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
