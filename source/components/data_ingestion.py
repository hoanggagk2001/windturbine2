import os 
import sys
import pyodbc
from source.exception import CustomException
from source.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from source.components.data_transformation import DataTransformation,DataTransformationConfig
from source.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str =os.path.join('artifacts', 'train.csv')
    raw_data_path:str =os.path.join('artifacts', 'data.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Enterd the data ingestion method or component")
        try:
            # server = r"NGOC-A-HOANG\NGUYENHOANG"
            # database = "Windturbine"
            # user = "windturbine"
            # password = "hoangga2001"

            # connection_string = "Driver={ODBC Driver 17 for SQL Server};" \
            #                     f"Server={server};" \
            #                     f"Database={database};" \
            #                     f"Uid={user};" \
            #                     f"Pwd={password};"

            # # Tạo kết nối
            # connection = pyodbc.connect(connection_string)

            # # print(pyodbc.drivers())
            # sqlQuery = "SELECT TOP(100000) *FROM dbo.data ORDER BY ID DESC "
            # df=pd.read_sql(sql=sqlQuery,con=connection)
            # df= df.sort_values(by="ID", ascending=True)
            # connection.close()
            # print(df)
            df=pd.read_csv('notebook\data\data.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df1 = df.loc[:,['PCTimeStamp','WTG01_Ambient WindSpeed Avg. (1)','WTG02_Ambient WindSpeed Avg. (2)','WTG03_Ambient WindSpeed Avg. (3)','WTG04_Ambient WindSpeed Avg. (4)','WTG05_Ambient WindSpeed Avg. (5)','WTG06_Ambient WindSpeed Avg. (6)','WTG07_Ambient WindSpeed Avg. (7)']]
            df1['PCTimeStamp']=pd.to_datetime(df1['PCTimeStamp'],format='mixed')
            df1 = df1.fillna(df1.groupby(df1.PCTimeStamp.dt.hour).transform('median'))
            df1.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Data ingestion completed")
            return(
                self.ingestion_config.train_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    trainX,trainY=data_transformation.initiate_data_transformation(train_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(trainX,trainY))

