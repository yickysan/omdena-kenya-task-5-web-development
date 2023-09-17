import sys
from pathlib import Path
from dataclasses import dataclass
from OmdenaKenyaRoadAccidents.utils import download_file_from_google_drive, split_save_data
from OmdenaKenyaRoadAccidents.logger import logging
from OmdenaKenyaRoadAccidents.exception import DataIngestionError


HERE = Path(__file__)
SRC_PATH = HERE.parent.parent # path to OmdenaKenyaRoadAccidents


@dataclass
class DataIngestionConfig:
    train_data_path: Path = SRC_PATH / "artifacts" / "train.csv"
    test_data_path: Path = SRC_PATH / "artifacts" / "test.csv"
    raw_data_path: Path = SRC_PATH / "artifacts" / "data.csv"



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, id: str) -> Path:
        logging.info("initialised the data ingestion method")
        try:
            # create directory for storing the data
            self.ingestion_config.raw_data_path.parent.mkdir(exist_ok=True, parents=True)

            # download the file from google drive
            download_file_from_google_drive(id=id,
                                             Path=self.ingestion_config.raw_data_path
                                             )
            logging.info("data was successfuly downloaded from google drive")
            
            

            logging.info("Train test split is initialised")

            data_path = (self.ingestion_config.raw_data_path,
                         self.ingestion_config.train_data_path,
                         self.ingestion_config.test_data_path
                         )
            split_save_data(data_path=data_path, test_size=0.2)

            logging.info("Data ingestion is completed")
            
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )

            

        except Exception as e:
            logging.info(f"error: {e}")
            raise DataIngestionError(e, sys)
        


