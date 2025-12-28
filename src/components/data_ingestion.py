import os 
import sys 
from src.exception import CustomException
from src.logger import logging # to use logging from the logger file in src
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.components.data_transformation import DataTransformationConfig
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass #decorator- adding more functionality to this already existing class
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train.csv")
    test_data_path=os.path.join('artifacts',"test.csv")
    raw_data_path=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()     #above three paths will get saved into ingestion_config

    def initiate_data_ingestion(self):
        logging.info("Entered the Data ingestion method or component")
        
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("Imported the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)      #here we make the above folders names the train test and raw data path folders 

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #saving the raw file 

            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)       #train-test split was done here

            train_set.to_csv(self.ingestion_config.train_data_path)       # data sent to train data
            test_set.to_csv(self.ingestion_config.test_data_path)         # data sent to test data

            logging.info("Ingestion of data and train test split DONE")

            return( #returning the train and test data 
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))