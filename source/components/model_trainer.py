import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras import Model, Sequential
import sys
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from dataclasses import dataclass
from keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell,Dropout
from source.exception import CustomException
from source.logger import logging
import os
from source.utils import save_object,model
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,trainX,trainY):
        self.trainX = trainX
        self.trainY = trainY

        try:
            model= Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(trainY.shape[1]))
            model.compile(optimizer='adam', loss='mse')
            model.fit(trainX, trainY, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
            logging.info('model doned')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
        except Exception as e:
            raise CustomException(e,sys)