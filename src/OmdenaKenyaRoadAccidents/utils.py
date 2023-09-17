import requests
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import typing
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from datetime import datetime


RESPONSE = requests.models.Response

def download_file_from_google_drive(id: str, Path: Path) -> None:
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, Path)    

def get_confirm_token(response: RESPONSE) -> str:
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response: RESPONSE, Path: Path):
    CHUNK_SIZE = 32768

    with open(Path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


RAW_DATA = Path
TRAIN_DATA = Path
TEST_DATA =  Path

def split_save_data(data_path: tuple[RAW_DATA, TRAIN_DATA, TEST_DATA], test_size: float) -> None:
    """
    Function to load csv as a pandas.DataFrame object and split into train and test set
    
    This function takes in the path to the data to be splitted, loads it as a pandas.DataFrame object
    and then uses scikit-learn.model_selection train_test_split fucntion to split the data before
    writing the train and test data back to csv

    Parameters
    ----------
    data_path : tuple[Path]
        a tuple containing the pathlib.Path object of the data to be loaded and the pathlib.Path object for saving the train and test data.
    
    test_size : float
        size of the test data when splitted using scikit-learn.model_selection train_test_split function.
    """

    raw_data_path, train_data_path, test_data_path = data_path
    df = pd.read_csv(raw_data_path)
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    
    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)


def save_object(path: Path, obj: typing.Any) -> None:
    """
    Function to save python objects as a pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean the dataframe.

    This function creates an hour column from the time column in the dataframe
    """

    return(df
           .assign(time = df.time.apply(lambda x: datetime.strptime(x, "%H:%M:%S").time().hour)
           )
    )


def select_features(df: pd.DataFrame, k: int=6) -> np.ndarray[str]:
    """
    Function to select top k features from the dataframe.
    
    This function uses scikit-learn.feature_selection SelectKBest fucntion and  chi2 scoring
    to select the top k features from the dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the features.

    k : int, optional, default = 6
        The number of features to be selected.
        
    Returns
    -------
    numpy.ndarray[str]
        A numpy array containg k number of features
    
    """
    cols = [col for col in df.columns if "casual" not in col]
    X = df[cols].copy().drop(["accident_severity", "time"], axis=1)
    y = df["accident_severity"]

    # convert string values in X to category data type
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes

    kbest = (SelectKBest(chi2, k=k)
             .fit(X, y)
             .get_feature_names_out())
    
    return kbest


class SklearnEstimator(typing.Protocol):
    def fit(self, X, y) -> None:
        ...

    def predict(self, X) -> np.ndarray:
        ...

REPORT = dict[dict[float]]
def evaluate_models(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray, 
                    models: dict[SklearnEstimator]) -> REPORT:
    """
    Function to evaluate models

    This function evaluates skearn models based on their accuracy score, their roc_auc_score,
    and their recall and precision score

    Parameters
    ----------
    X_train : numpy.ndarray
        a numpy array of the train dataset containing the features
    
    y_train : numpy.ndarray
        a numpy array containing the target variables for the train dataset
    
    X_test : numpy.ndarray
        a numpy array of the t4st dataset containing the features
    
    y_test : numpy.ndarray
        a numpy array containing the target variables for the test dataset
    
    models : dict[SklearnEstimator]
        a dictionary containg models that implements scikit-learn fit and predict methods

    
    Returns
    -------
    REPORT : dict[dict[float]]
        a nested dictionary containing scikit-learn.metrics accuracy_score, roc_auc_score,
        recall_score and precision_score for the different models
    """

    report = {}

    for k, v in models.items():
        scores = {}
        model_name = k
        model = v

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        # roc_auc_score needs the probablilites to work for multi class
        auc_score = roc_auc_score(y_test, proba, multi_class="ovr")
        recall = recall_score(y_test, preds, average="weighted")
        precision = precision_score(y_test, preds, average="weighted")

        scores["accuracy"] = accuracy
        scores["auc_score"] = auc_score
        scores["recall"] = recall
        scores["precision"] = precision

        report[model_name] = scores

    return report


def load_artifacts() -> dict[typing.Any]:
    """
    Function to load model and preprocessor
    """
    # gets the precise directory of the artifacts directory
    ARTIFACTS = Path(__file__).parent / "artifacts"

    model_path = ARTIFACTS / "model.pkl"
    X_preprocessor_path = ARTIFACTS / "x_preprocessor.pkl"
    y_preprocessor_path = ARTIFACTS / "y_preprocessor.pkl"

    paths = (model_path, X_preprocessor_path, y_preprocessor_path)
    names = ["model", "X_preprocessor", "y_preprocessor"]

    artifacts = {}

    for path, name in zip(paths, names):
        with open(path, "rb") as f:
            artifacts[name] = pickle.load(f)

    return artifacts


# def get_color(result: str) -> str:
#     """
#     Function to return hex color based on prediction results
#     """
#     color_dict= {"Serious Injury": "#db8b23", 
#                  "Slight Injury": "#06630c", 
#                  "Fatal injury": "#db2f23"}
    
#     return color_dict[result]



