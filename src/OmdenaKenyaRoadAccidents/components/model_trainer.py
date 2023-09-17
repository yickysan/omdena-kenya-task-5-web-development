import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, accuracy_score, precision_score

from OmdenaKenyaRoadAccidents.exception import ModelTrainerError
from OmdenaKenyaRoadAccidents.logger import logging
from OmdenaKenyaRoadAccidents.utils import save_object, evaluate_models, SklearnEstimator

HERE = Path(__file__)
SRC_PATH = HERE.parent.parent

@dataclass
class ModelTrainerConfig:
    model_file_path : Path = SRC_PATH / "artifacts" / "model.pkl"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr: np.ndarray,
                                test_arr: np.ndarray,
                                model: Optional[SklearnEstimator]=None) -> tuple[float, float, float, float,  np.ndarray]:
        try:
            logging.info("Splitting the training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], # takes every row and column except for the last column
                train_arr[:,-1], # takes every row in the last column
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # if a model is giving it is used to fit the data 
            # else the predifined models are used
            if model:
                best_model = model
                logging.info(f"Best Model: {best_model}")

                best_model.fit(X_train, y_train)

                logging.info("saving best model")

                save_object(
                    path=self.model_trainer_config.model_file_path,
                    obj=best_model
                )

            else:

                # training models
                logging.info("initialising different models")
                models = {}

                models["random_forest"] = RandomForestClassifier(class_weight="balanced", random_state=1)
                models["log_reg"] = LogisticRegression(solver="newton-cg", class_weight="balanced")
                models["svm"] = SVC(class_weight="balanced", random_state=1, probability=True)

                logging.info(f"inintialised models: {[models[model] for model in models]}")

                model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

                logging.info("model training complete")
                logging.info("initialising best model")

                # best performing model is picked based on the auc score rather than accuracy
                # because the data is imbalanced
                best_model = models[
                    sorted(models, key= lambda model: model_report[model]["auc_score"], reverse=True)[0]
                    ]
                
                logging.info(f"Best model: {best_model}")

                best_model.fit(X_train, y_train)

                logging.info("saving best model")

                save_object(
                    path=self.model_trainer_config.model_file_path,
                    obj=best_model
                )


            predictions = best_model.predict(X_test)
            proba = best_model.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, proba, multi_class="ovr")
            con_matrix = confusion_matrix(y_test, predictions)
            precision = precision_score(y_test, predictions, average="weighted")
            recall = recall_score(y_test, predictions, average="weighted")
            accuracy = accuracy_score(y_test, predictions)

            return(
                auc_score,
                recall,
                precision,
                accuracy,
                con_matrix
            )

        except Exception as e:
            logging.info(f"error: {e}")
            raise ModelTrainerError(e, sys)