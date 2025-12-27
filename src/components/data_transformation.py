import os 
import sys

from src.logger import logging
from src.exception import CustomException #imported logger and exception from our other files so that we can use it here
from src.utils import save_object #this is the process of bringing in a function we wrote inside the utils file to use here 
from dataclasses import dataclass

import numpy as np   # to create array and all for the ml model
import pandas as pd  # to read csv


from sklearn.preprocessing import OneHotEncoder,StandardScaler  #used for encoding and changing into numerical data, and then scaling it to a range/ Normalization of values
from sklearn.compose import ColumnTransformer  #used to apply pipelilne to certain arrays or columns 
from sklearn.impute import SimpleImputer  #used to fill in missing values
from sklearn.pipeline import Pipeline  #to create pipeline and put steps in an order


@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts',"Preprocessor.pkl")  #adding the preprocessor pickle file into the artifacts directory

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() #initializing the data tranformation config as the above function

    def get_data_transformer_object(self):

        try:
            numerical_columns=['reading_score', 'writing_score'] #numerical columns we got during eda
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'] #categorial columns we got during eda

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),   #handles missing values
                    ("Scaler",StandardScaler())            #scales the above values
                ]
            )


            logging.info("Numerical columns scaling and imputation completed, no encoding was done here")


            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),      #replace with mode basically
                    ("one hot encoder",OneHotEncoder()),                    #encodes into numerical values as we studied
                    ("scaler",StandardScaler(with_mean=False))            #scales those values we got earlier

                ]
            )

            logging.info("Categorial columns imputation, encoding and scaling completed")

            preprocessor=ColumnTransformer(
                [
                    ("Numerical_Pipeline",num_pipeline,numerical_columns),      #applying the Numerical pipeline functionality onto the Numerical columns we have
                    ("Categorial_Pipeline",cat_pipeline,categorical_columns)   #applying the Categorial pipeline functionality onto the Categorial columns we have
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path) #putting train path csv into train df
            test_df=pd.read_csv(test_path)   #putting test path csv into test df

            logging.info("Reading train and test data completed")  #basic logging

            logging.info("Obtain preprocessing object")   #basic logging

            preprocessor_obj=self.get_data_transformer_object()  #bringing the preprocessor here

            target_column_name="math_score"
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying preprocessing on train and test dataframe")

            input_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_test_arr,np.array(target_feature_test_df)]

            logging.info("Concatenated input features to the output feature in the final feature matrix")

            logging.info("Saving the preprocessing to the pickle file (pkl) to use in the future")

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
                
            )
        


        except Exception as e:
            raise CustomException(e,sys)




