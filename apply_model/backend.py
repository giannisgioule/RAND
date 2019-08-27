import pandas as pd
import numpy as np
import joblib

import os

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# The backend for regression methods.
class BackendRegression():

    def __init__(self,context):
        self.context=context
    
    
    def backend(self):
        
        method=self.context["method"]
        approach=self.context["approach"]
        dataset=self.context["dataset"]
        hyperparameters=self.context["hyperparameters"] 
        approach_parameters=self.context["approach_parameters"]


        def read_file(input_data):
            data=pd.read_csv(input_data)
            number_predictors=data.shape[1]-1
            variable_names=data.columns[0:number_predictors]
            X=data.iloc[:,0:number_predictors]
            y=data.iloc[:,(data.shape[1]-1)]                                    

            return X,y

        def train_model(X,y,method,hyperparameters):                                

            if method=="linear_regression":
                model=LinearRegression(**hyperparameters)                            
            elif method=="decision_tree":
                model=DecisionTreeRegressor(**hyperparameters)                        
            elif method=="mlp":
                model=MLPRegressor(**hyperparameters)                                    
            elif method=="random_forest":
                model=RandomForestRegressor(**hyperparameters)                                
            elif method=="svr":
                model=svm.SVR(**hyperparameters)                                    

            model.fit(X,y)            
            
            return model
        
        def validate_model(X,y,method,approach,approach_parameters):
            
            if approach=="train_test":
                X_train,X_test,y_train,y_test=train_test_split(X,y,**approach_parameters)                                    
                return X_train,X_test,y_train,y_test
            if approach=="kfold":
                kf=KFold(**approach_parameters)
                index_list=kf.split(X)
                return index_list
        
        def calculate_metrics(y_true,y_pred):
            
            metrics={
                "mae":round(mean_absolute_error(y_true=y_true,y_pred=y_pred,\
                    multioutput="uniform_average"),3),
                "mse":round(mean_squared_error(y_true=y_true,y_pred=y_pred,\
                    multioutput="uniform_average"),3),
                "ex_variance":round(explained_variance_score(y_true=y_true,\
                    y_pred=y_pred,multioutput="uniform_average"),3),
                "r2":round(r2_score(y_true=y_true,y_pred=y_pred,\
                    multioutput="uniform_average"),3)
            }

            return metrics

        X,y=read_file(dataset)

        if approach=="full_training":
            model=train_model(X,y,method,hyperparameters)  

            y_pred=model.predict(X)
            metrics_training=calculate_metrics(y,y_pred)

        elif approach=="train_test":
        
            X_train,X_test,y_train,y_test=validate_model(X,y,method,approach,\
                approach_parameters)            

            model=train_model(X_train,y_train,method,hyperparameters)
            y_pred=model.predict(X_test)
            y_pred_train=model.predict(X_train)

            metrics_training=calculate_metrics(y_train,y_pred_train)
            metrics_testing=calculate_metrics(y_test,y_pred)
            
            self.metrics_testing=metrics_testing                
        
        elif approach=="kfold":
            
            index_list=validate_model(X,None,method,approach,\
                approach_parameters)
            
            metrics_training=[]
            metrics_testing=[]

            for train_index, test_index in index_list:
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]

                model=train_model(X_train,y_train,method,hyperparameters)
                y_pred=model.predict(X_test)
                y_pred_train=model.predict(X_train)
                
                metrics_training.append(list(calculate_metrics(y_train,y_pred_train).values()))
                metrics_testing.append(list(calculate_metrics(y_test,y_pred).values()))
                
            metrics_training=np.transpose(metrics_training)
            metrics_testing=np.transpose(metrics_testing)
            
            metrics_training=[round(np.average(x),3) for x in metrics_training]
            metrics_testing=[round(np.average(x),3) for x in metrics_testing]

            metrics_training={
                "mae":metrics_training[0],
                "mse":metrics_training[1],
                "ex_variance":metrics_training[2],
                "r2":metrics_training[3]
                }
            
            metrics_testing={
                "mae":metrics_testing[0],
                "mse":metrics_testing[1],
                "ex_variance":metrics_testing[2],
                "r2":metrics_testing[3]
                }

            self.metrics_testing=metrics_testing                

        self.model=model
        self.metrics_training=metrics_training
        
        return self

    def predict(self):
        
        model=self.context["model"]
        dataset=self.context["dataset"]
        
        def read_file(input_data):
            data=pd.read_csv(input_data)
            number_predictors=data.shape[1]            
            X=data.iloc[:,0:number_predictors]            

            return X
        
        X=read_file(dataset)
        y_pred=model.predict(X)                               
        if os.path.exists("apply_model/static/apply_model/predictions.csv"):
            os.remove("apply_model/static/apply_model/predictions.csv")
        text_file=open("apply_model/static/apply_model/predictions.csv","w+")
        [text_file.write("{}\n".format(round(x,3))) for x in y_pred]

        self.predictions=y_pred
        return self            