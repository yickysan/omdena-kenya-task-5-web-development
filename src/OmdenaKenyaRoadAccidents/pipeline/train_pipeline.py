from OmdenaKenyaRoadAccidents.components.data_ingestion import DataIngestion
from OmdenaKenyaRoadAccidents.components.data_transformation import DataTransformation
from OmdenaKenyaRoadAccidents.components.model_trainer import ModelTrainer
from OmdenaKenyaRoadAccidents.utils import SklearnEstimator
from typing import Optional
from collections import namedtuple

RESULT = namedtuple("results", ["roc_auc_score", "recall_score",
                                "precision_score", "accuracy_score",
                                "confusion_matrix"]
                                )

def train_pipeline(id: str, model: Optional[SklearnEstimator]=None) -> RESULT:
    """
    Function to create a pipeline from data ingestion to model training

    This function uses the DataIngestion, DataTransformation and ModelTrainer
    classes and performs the following steps:
    1: Download the data from google drive using the file id provided
    2: Split the data into train and test data and store them in the given path
    3: Transform the data using `scikit-learn.preprocessing` OneHotEncoder for the
       features variables and LabelEncoder for the target variable.
    4: Train a model with the data using the model provided with the model argument
       or with the already preloaded models in the ModelTrainer class.

    Parameters
    ----------
    id : str
        google drive file id
    
    model : SklearnEstimator, optional, default = None
        machine learning estimator that implements the `scikit-learn` fit and predict methods
    
    Returns
    -------
    results : namedtuple[roc_auc_score, recall_score, 
                    precision_score, accuracy_score,
                    confusion_matrix]
        tuple containing `scikit-learn.metrics` evaluation metrics for classification
    """
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion(id=id)
    print("Data Ingestion Completed!")

    data_transformer = DataTransformation()
    train_array, test_array = data_transformer.initiate_data_transformation(train_path, test_path)
    print("Data Transormation Completed!")

    
    model_trainer = ModelTrainer()
    auc_score, recall, precision, accuracy, con_matrix = model_trainer.initiate_model_trainer(
        train_array, test_array, model=model)
    print("Model Training Compeleted")

    return RESULT(auc_score, recall, precision, accuracy, con_matrix)

    


    