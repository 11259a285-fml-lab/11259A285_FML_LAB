from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
iris  = load_iris()
X = iris.data
y = iris.target
print("Dataset shape:",X.shape)
print("Number of samples :",len(X))
print("Number of features:",X.shape[1])
print("target names:",iris.target_names)
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.20,
random_state= 42,
stratify = y
)
print("Dataset splitting")
print("--------------")
print("Training samples:",X_train)
print("Testing samples:",X_test)
model = LogisticRegression(max_iter = 200)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print("Test set performance")
print("-------------")
print("Accuracy:",round(accuracy,2))
kf = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42,
)
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv  = kf,
    scoring = "accuracy"
)
print('cross validation')
print("---------------")
for i,score in enumerate(cv_scores,start = 1):
    print("Fold",i,"Accuracy:",round(score,4))
    print("Cross validation score")
    print("---------------")
    print("Average accuracy:",round(cv_scores.mean(),4))
    print("Standard deviation:",round(cv_scores.std(),4))










