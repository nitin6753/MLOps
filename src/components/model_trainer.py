import os
import sys
from src.exception import customException
from src.logger import logging

from dataclasses import dataclass
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_model
from src.components.model_selection import get_model

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split train and test input array into feature and target")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1],
                                                test_array[:,:-1], test_array[:,-1])
            
            model_name = [
                "Random Forest", "Decision Tree",
                "Gradient Boosting", "Linear Regression",
                "XGB Regressor", "CatBoosting Regressor",
                "AdaBoost Regressor", "KNN Regressor"
            ]
            
            models = get_model(model_name)
            
            model_report, gscv_model = evaluate_model(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test,
                                                models=models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = gscv_model[best_model_name]

            if best_model_score < 0.6:
                raise customException("All Models have r2_score less than 0.6")

            logging.info(f"Model training finished, best model found: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Saved model: {best_model_name}")

            y_preds = best_model.predict(X_test)
            r2 = r2_score(y_test,y_preds)
            logging.info(f"R2 score of {best_model_name}: {r2}")
            
            return r2

        except Exception as e:
            raise customException(e,sys)