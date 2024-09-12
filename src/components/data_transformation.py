import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import customException
from src.logger import logging
from src.utils import save_object, get_columns

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,numerical_columns,categorical_columns):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical columns transformed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns transformed")

            preprocessor = ColumnTransformer(
                [("numerical_pipeline",num_pipeline,numerical_columns),
                ("categorical_pipeline",cat_pipeline,categorical_columns)]
            )

            return preprocessor
        
        except Exception as e:
            raise customException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test dataset reading completed")
            
            target_column = 'math_score'
            numerical_columns, categorical_columns = get_columns(train_df,target_column)
            logging.info(f"Target Variable: {[target_column]}")
            logging.info(f"Numerical Features: {numerical_columns}")
            logging.info(f"Categorical Features: {categorical_columns}")
            
            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info(f"Target variables and features extracted from training and test datasets")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object(numerical_columns,categorical_columns)
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Applied preprocessing object on the train and test datasets")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info(f"Saved preprocessing object")

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise customException(e,sys)
        
