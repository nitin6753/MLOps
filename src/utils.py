import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import customException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)    
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
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