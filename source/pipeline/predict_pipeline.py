import sys
import pandas as pd
from source.exception import CustomException
from source.utils import load_object
import os
from source.components.data_transformation import DataTransformation,DataTransformationConfig
from source.components.model_trainer import ModelTrainer,ModelTrainerConfig
from source.components.data_ingestion import DataIngestion

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,n_fultures):
        
        self.n_fultures = n_fultures
        try:
            # train_df=pd.read_csv(train_path)
            # train_dates=pd.to_datetime(train_df['PCTimeStamp'],format='mixed')
            # cols=list(train_df)[1:8]
            # df_for_training=train_df[cols].astype(float)
            model_path=os.path.join('artifacts',"model.pkl")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            obj=DataIngestion()
            train_data=obj.initiate_data_ingestion()

            data_transformation=DataTransformation()
            trainX,trainY=data_transformation.initiate_data_transformation(train_data)
            pred=model.predict(trainX[-n_fultures:])
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        n_fultures :int):
        self.n_fultures= n_fultures
    def get_data(self):
        try:
            custom_data_input_dict={
                "n_fultures":self.n_fultures
            }
            nfulture=custom_data_input_dict["n_fultures"]
            return nfulture
        except Exception as e:
            raise CustomException(e,sys)
    
