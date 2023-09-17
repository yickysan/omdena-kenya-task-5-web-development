import pandas as pd
import numpy as np
from OmdenaKenyaRoadAccidents.utils import load_artifacts

class PredictPipeline:
    def __init__(self,
                 age_band_of_driver: str,
                 driving_experience: str,
                 defect_of_vehicle: str,
                 light_conditions: str,
                 number_of_vehicles_involved: str,
                 cause_of_accident: str):
        
        self.age_band_of_driver = age_band_of_driver
        self.driving_experience = driving_experience
        self.defect_of_vehicle = int(defect_of_vehicle)
        self.light_conditions = light_conditions
        self.number_of_vehicles_involved = int(number_of_vehicles_involved)
        self.cause_of_accident = cause_of_accident


    def _create_df(self) -> pd.DataFrame:
        df = pd.DataFrame([
            {"age_band_of_driver": self.age_band_of_driver,
             "driving_experience": self.driving_experience,
             "defect_of_vehicle": self.defect_of_vehicle,
             "light_conditions": self.light_conditions,
             "number_of_vehicles_involved": self.number_of_vehicles_involved,
              "cause_of_accident": self.cause_of_accident
              }]
              )
        
        return df
    
    def predict(self) -> np.ndarray[str]:
        self.artifacts = load_artifacts()
        df = self._create_df()
        X = self.artifacts["X_preprocessor"].transform(df)
        pred = self.artifacts["model"].predict(X)
        result = self.artifacts["y_preprocessor"].inverse_transform(pred.astype(int))

        return result