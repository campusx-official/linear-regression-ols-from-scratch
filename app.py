from sklearn import datasets
from LinearRegression import MyLR

X,y = datasets.load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
#reg = LinearRegression()
reg1 = MyLR()

reg1.fit(X_train,y_train)

y_pred = reg1.predict(X_test)

#reg.fit(X_train,y_train)

#y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#print(reg.coef_)
#print(reg.intercept_)

