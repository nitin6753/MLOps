import os
import sys
import numpy as np
import pandas as pd
import dill

# Load Models
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,accuracy_score,recall_score,precision_score,f1_score,mean_squared_error

from src.exception import customException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)    
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise customException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise customException(e,sys)

def get_columns(df,target_column):
    '''
    Returns numerical columns:Index, categorical columns:Index
    with target column removed to transform features
    '''
    try:
        numerical_columns = df.select_dtypes(exclude="object").columns
        categorical_columns = df.select_dtypes(include="object").columns
        
        if target_column in numerical_columns:
            numerical_columns=numerical_columns.drop(target_column)
        if target_column in categorical_columns:
            categorical_columns=categorical_columns.drop(target_column)
        
        return numerical_columns,categorical_columns
    
    except Exception as e:
        raise customException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        best_estimators={}

        for i in range(len(list(models))):
            
            model_name = list(models.keys())[i]
            model = models[model_name]['model']
            scoring = ['r2','neg_mean_squared_error']
            params = models[model_name]['params']

            gscv = GridSearchCV(model,params,scoring=scoring,cv=3,refit='r2')
            gscv.fit(X_train,y_train)
            model = gscv.best_estimator_
            best_estimators[model_name]=model
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
            logging.info(f"Model Trained: {model_name}")

        return report, best_estimators
    
    except Exception as e:
        raise customException(e,sys)

def get_model(model_name):
    try:
        avail_models = avail_model_list()
        select_model={}
        for i in model_name:
            select_model[i]=avail_models[i]
        logging.info("All Models are selected")
        return select_model
    except Exception as e:
        raise customException(e,sys)
    
def avail_model_list():
    models={
        "Random Forest":{
            'model':RandomForestRegressor(),
            'params':{
                # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,64,128,256]
            }
        },
        "Decision Tree": {
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            }
        },
        "Gradient Boosting":{
            'model':GradientBoostingRegressor(),
            'params':{
                # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            }
        },
        "Linear Regression":{
            'model':LinearRegression(),
            'params':{

            }
        },
        "KNN Regressor":{
            'model':KNeighborsRegressor(),
            'params':{
                
            }
        },
        "XGB Regressor":{
            'model':XGBRegressor(),
            'params':{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
        },
        "CatBoosting Regressor":{
            'model':CatBoostRegressor(verbose=False,train_dir='artifact/catboost_info/'),
            'params':{
                'depth': [6,8,10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            }
        },
        "AdaBoost Regressor":{
            'model':AdaBoostRegressor(),
            'params':{
                'learning_rate':[.1,.01,0.5,.001],
                # 'loss':['linear','square','exponential'],
                'n_estimators': [8,16,32,64,128,256]
            }
        }
    }
    return models