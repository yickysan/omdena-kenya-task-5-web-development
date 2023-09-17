# Predicting Road Accident Severity
This is a project which I built while collaborating with Omdena Kenya local chapter. My task was to build a machine learning web app and deploy a machine learning model for predicting road accidents.

## Project Goals
* This project aims to leverage machine learning (ML) techniques to analyze and predict road accidents in Kenya.
* Analyze and understand the patterns and contributing factors of accidents on Kenyan roads using * historical accident data.
* Identify accident-prone areas (hotspots) by analyzing accidents' spatial and temporal patterns.
* Develop a machine learning model to predict the severity of accidents based on various factors such as road conditions, weather, time of day, and vehicle types.

## Requirements
* pandas
* numpy
* plotly
* matplotlib
* seaborn
* scikit-learn
* notebook
* flask

## Run
To run the project you need to install the dependencies using `pip install -r requirements.txt`
This is going to build the OmdenaKenyaRoadAccidents package in your local machine. 

There are 5 major components:
* DataIngestion class
* DataTransformation class
* ModelTrainer class
* train_pipeline function
* PredictPipeline class

These components are used to download the data from Google drive, apply preprocessing, add a machine learning model and then make predictions from the form input coming from the web app. 

The DataIngestion class `initiate_data_ingestion` method is used to download the data from Google drive. This method takes one argument `id` which is the file id of the data to be downloaded. The downloaded file is the split into a train and test data set, with the test data being 20% of the original dataset. The datasets are stored in the artifacts directory. 

#### data ingestion example
```python
from OmdenaKenyaRoadAccidents.components.data_ingestion import DataIngestion

id = "sYHT1jdjdieiW?ejieX3"
data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion(id=id)
```

The DataTransformation class has the method `initiate_data_transformation` which takes the path to the train and test data as arguments. This method uses `scikit-learn` SelectKBest class to select the top features and then uses `scikit-learn` OneHotEnconder preprocessor to transform the features variable and LabelEncoder to transform the target variable. It returns a numpy array of the train and test data. Both preprocessor objects are stored in the artifacts directory. 

#### data transfornation example
```python
from OmdenaKenyaRoadAccidents.components.data_transformation import DataTransformation

data_transformer = DataTransformation()
train_array, test_array = data_transformer.initiate_data_transformation(train_path, test_path)
```

The ModelTrainer class like the others has an `initiate_model_trainer` method which takes a train_arr argument which is a numpy array of the train data, a test_arr which is a numpy array of the test data and an optional model argument which is the model that will be used to train the data. If no model is given it selects the best out of a `scikit-learn` LogisticRegression model, a RandomForestClassifier model an SVC model, the best model is selected based on their `roc_auc_score` since the data is an imbalanced data. The method returns the following `scikit-learn` metrics -roc_auc_score, recall_score, precision_score, accuracy_score and confusion_matrix

#### model trainer example
```python
from OmdenaKenyaRoadAccidents.components.model_trainer import ModelTrainer

model_trainer = ModelTrainer()
auc_score, _,_,_,_ = model_trainer.initiate_model_trainer(train_array, test_array) 

```

OR

```python
from OmdenaKenyaRoadAccidents.components.model_trainer import ModelTrainer
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(class_weight="balanced")
model_trainer = ModelTrainer()
auc_score, _,_,_,_ = model_trainer.initiate_model_trainer(train_array, test_array, model=model)

```

The train_pipeline function stitches the entire training process together. the function takes two arguments, an `id` which is the file id argument required by the DataIngestion class, and an optional model argument required by the model trainer class. this fucntion is called in the main module and is used to activate the afformentioned steps performed by the three classes. The function returns a named tuple containing the afformentioned `scikit-learn` metrics

#### train pipeline example
```python
from OmdenaKenyaRoadAccidents.pipeline.train_pipeline import train_pipeline

from sklearn.tree import DecisionTreeClassifier

def main() -> None:
    id = "sYHT1jdjdieiW?ejieX3"

    model = DecisionTreeClassifier(class_weight="balanced")

    results = train_pipeline(id=id, model=model)

    print(results)


if __name__ == "__main__":
    main()
```
The PredictPipeline class gets data from the form in the web app and is used to make predictions on the web app.
```python
@app.route("/predictions", methods=["GET", "POST"])
def make_prediction():
    if request.method == "GET":
        return render_template("predict.html")
    
    else:
                
        pipeline = PredictPipeline(age_band_of_driver = request.form.get("driver-age"),
                                   driving_experience = request.form.get("driving-experience"),
                                   defect_of_vehicle = request.form.get("vehicle-defect") ,
                                   light_conditions = request.form.get("light-condition"),
                                   number_of_vehicles_involved = request.form.get("no-of-vehicles-involved"),
                                   cause_of_accident = request.form.get("cause-of-accident")
                                   )
                

        result = pipeline.predict()[0]

```
