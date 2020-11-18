import numpy as np

class MyLR:

    def __int__(self):
        self.coef_=None
        self.intercept_ = None

    def fit(self,X_train,y_train):
        X = []
        for row in X_train:
            row = np.insert(row,0,1)
            X.append(row)
        X = np.array(X)

        beta = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y_train)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        print(self.intercept_)
        print(self.coef_)

    def predict(self,X_test):

        y_pred = []

        for row in X_test:
            y_pred.append(np.dot(row,self.coef_) + self.intercept_)

        return np.array(y_pred)
