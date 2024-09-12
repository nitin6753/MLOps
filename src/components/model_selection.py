import os
import sys
from src.exception import customException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def get_model(model_name):
    avail_models = avail_model_list()
    select_model={}
    for i in model_name:
        select_model[i]=avail_models[i]
    return select_model

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
            'params':{}
        },
        "KNN Regressor":{
            'model':KNeighborsRegressor(),
            'params':{}
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