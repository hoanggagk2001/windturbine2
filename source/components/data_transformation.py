import sys
from dataclasses import dataclass
import datetime
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from source.exception import CustomException
from source.logger import logging
import os
from source.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # def get_data_transformer_object(self):
    #     '''
    #     This function is responsible for data transformation
        
    #     '''
    #     try:
    #         data_columns=["WTG01_Ambient WindSpeed Avg. (1)","WTG02_Ambient WindSpeed Avg. (2)","WTG03_Ambient WindSpeed Avg. (3)","WTG04_Ambient WindSpeed Avg. (4)","WTG05_Ambient WindSpeed Avg. (5)","WTG06_Ambient WindSpeed Avg. (6)","WTG07_Ambient WindSpeed Avg. (7)"]
    #         data_pipeline=Pipeline(
    #             steps=[
    #             ('imputer', SimpleImputer(strategy='median')),
    #             ('scaler', MinMaxScaler())
    #         ]
    #         )
    #         preprocessor=ColumnTransformer(
    #             [
    #             ("data_pipeline",data_pipeline,data_columns)

    #             ]
    #         )
    #         return preprocessor
    #     except Exception as e:
    #         raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path):

        try:
            train_df=pd.read_csv(train_path)
            logging.info("Read train data completed")
            # preprocessing_obj=self.get_data_transformer_object()
            #Separate dates for future plotting
            train_dates=pd.to_datetime(train_df['PCTimeStamp'],format='mixed')
            cols=list(train_df)[1:8]
            df_for_training=train_df[cols].astype(float)
            scaler=MinMaxScaler()
            scaler=scaler.fit(df_for_training)
            df_for_training_scaled=scaler.transform(df_for_training)
            # df_for_training_scaled=preprocessing_obj.fit_transform(df_for_training)
            trainX=[]
            trainY=[]
            n_future=1 #Number of hour we want to predict into the future
            n_past=7 # Number of past day we want  to use to predict the future

            for i in range(n_past,len(df_for_training_scaled)-n_future+1):
                trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
                trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
            trainX, trainY = np.array(trainX), np.array(trainY)
            logging.info("Data transformation completed")
            
            return (
                trainX,
                trainY
                
            )
        except Exception as e:
            raise CustomException(e,sys)


