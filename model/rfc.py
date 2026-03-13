from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class RFC(RandomForestClassifier):
    def __init__(self,X,Y,n_estimators,max_depth,max_features,random_state,min_samples_leaf,min_samples_split,oob_score:bool):
        super().__init__(n_estimators=n_estimators,random_state=random_state,max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,oob_score=oob_score)
        self.random_seed=random_state
        self.X=X
        self.Y=Y
        self.oob_score=oob_score
        # self.lasso_result=None
        # self.split_data=split_data
        self.lasso_result=np.array([99999]*len(self.X.columns))
        self.col_rm = self.X.columns[np.abs(self.lasso_result) < 0.001]
    
    def _split_dataset(self,test_size=None):
        if self.oob_score==False:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=self.random_seed)
            # return X_train, X_test, y_train, y_test
            self.X_train=X_train
            self.Y_train=Y_train

            self.X_test=X_test
            self.Y_test=Y_test
        else:
            self.X_train=self.X
            self.Y_train=self.Y
            self.X_test=None
            self.Y_test=None
    def _get_external_splited_dataset(self,X_train,Y_train,X_test,Y_test):
        if self.oob_score==False:
            self.X_train=X_train
            self.Y_train=Y_train
            self.X_test=X_test
            self.Y_test=Y_test
        else:
            self.X_train=self.X
            self.Y_train=self.Y
            self.X_test=None
            self.Y_test=None
    
    def _lasoo(self,alpha,threshold):
        
        lasso = Lasso(alpha=alpha)  # alpha是正则化强度参数
        lasso.fit(self.X_train, self.Y_train)
        cols_to_remove = self.X_train.columns[np.abs(lasso.coef_) < threshold]
        self.X_train = self.X_train.drop(columns=cols_to_remove)
        try:
            self.X_test = self.X_test.drop(columns=cols_to_remove)
        except:
            pass
        self.lasso_result=lasso.coef_
        self.col_rm=cols_to_remove
        # self.threshold=threshold
        # return lasso.coef_

    def _fit_by_train_dataset(self):
        result=super().fit(X=self.X_train,y=self.Y_train)
        return result
    def _predict_test_dataset(self):
        pass

    def _cal_accuracy_test(self):
        self.Y_test_predict = super().predict(self.X_test)
        self.accuracy_test = accuracy_score(self.Y_test, self.Y_test_predict)

    def _cal_accuracy_train(self):
        self.Y_train_predict=super().predict(X=self.X_train)
        self.accuracy_train=accuracy_score(self.Y_train, self.Y_train_predict)

    def _get_report(self):
        return classification_report(self.Y_test, self.Y_test_predict)
    
    def _get_confusion_matrix(self):
        return confusion_matrix(self.Y_test, self.Y_test_predict)
    
    def _predict_after_lasso(self,X):
        # cols_to_remove = self.X_train.columns[np.abs(self.lasso_result) < self.threshold]
        X = X.drop(columns=self.col_rm)
        return super().predict(X=X)
    
    def _print_status(self):
        for attr, value in self.__dict__.items():
            if 'X' not in attr and 'Y'not in attr:
                print(f"{attr}: {value}")
    def _predict_proba_after_lasso(self,X):
        X = X.drop(columns=self.col_rm)
        return super().predict_proba(X=X)
    
if __name__=='__main__':
    # rfc=RFC(X=1,Y=1,n_estimators=100,max_depth=1,max_features=7,random_state=2)
    # print(dir(rfc))
    # print(rfc.n_jobs,rfc.n_estimators)
    pass