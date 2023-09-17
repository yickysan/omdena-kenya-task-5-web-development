import sys
from pathlib import Path
from dataclasses import dataclass 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from OmdenaKenyaRoadAccidents.utils import save_object, clean_df, select_features
from OmdenaKenyaRoadAccidents.exception import DataTransformationError
from OmdenaKenyaRoadAccidents.logger import logging

HERE = Path(__file__)
SRC_PATH = HERE.parent.parent

@dataclass
class DataTransformationConfig:
    X_preprocessor_path = SRC_PATH / "artifacts" / "X_preprocessor.pkl"
    y_preprocessor_path = SRC_PATH / "artifacts" / "y_preprocessor.pkl"

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _get_data_transformer_object(self):
        try:
            logging.info("initialising X and y preprocessor objects")

            X_peprocessor = OneHotEncoder(sparse_output=False)
            y_preprocessor = LabelEncoder()

            return X_peprocessor, y_preprocessor
        
        except Exception as e:
            raise DataTransformationError(e, sys)

    def initiate_data_transformation(self, 
                                     train_path: Path, 
                                     test_path: Path) -> tuple[np.ndarray, np.ndarray, Path, Path]:
        try:
            train_df = clean_df(
                pd.read_csv(train_path)
            )
            test_df = clean_df(
                pd.read_csv(test_path)
            )


            logging.info("Reading of train and test dataset completed.")

            logging.info("Selecting Best Features")

            best_features = select_features(train_df)

            logging.info(f"Best features: {best_features}")


            logging.info("Initialising preprocessor object.")
            
            X_preprocessor, y_preprocessor = self._get_data_transformer_object()
            

            target = "accident_severity"
            
            train_features = train_df[best_features]
            train_target = train_df[target]

            test_features = test_df[best_features]
            test_target = test_df[target]

            logging.info("Applying preprocessing on the train and test dataset")

            train_features_arr = X_preprocessor.fit_transform(train_features)
            test_features_arr = X_preprocessor.transform(test_features)

            train_target_arr = y_preprocessor.fit_transform(train_target)
            test_target_arr = y_preprocessor.transform(test_target)

            # concatenate features and target arrays
            train_arr = np.c_[
                train_features_arr, train_target_arr
                ]
            

            test_arr = np.c_[
                test_features_arr, test_target_arr
                ]
            
            logging.info("Saving preprocessor objects")

            save_object(
                path = self.data_transformation_config.X_preprocessor_path,
                obj = X_preprocessor
            )

            save_object(
                path = self.data_transformation_config.y_preprocessor_path,
                obj = y_preprocessor
            )
            
            return (train_arr, test_arr)
            

        except Exception as e:
            logging.info(f"error: {e}")
            raise DataTransformationError(e, sys)
