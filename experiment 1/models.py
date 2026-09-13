from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking,LSTM, Dense
from sklearn.svm import SVC


class Models:

    def __init__(self, n_estimators=100, epochs=10, batch_size=32, rnn_units=150, n_timesteps=100 ,n_features=36, padding_value=0 ):
        self.rf_estimators=n_estimators
        self.timesteps = n_timesteps
        self.features = n_features
        self.rnn_units = rnn_units
        self.padding_value = padding_value

    def RandomForest(self):
        self.model = RandomForestClassifier(n_estimators=self.rf_estimators)
        return self.model


    def SVM(self):
        self.model = SVC(kernel='poly',degree=3, random_state=42)
        return self.model
    
    def RNN(self):
        self.model = Sequential()
        self.model.add(Masking(mask_value=self.padding_value, input_shape=(self.timesteps, self.features)))
        self.model.add(LSTM(self.rnn_units,return_sequences=True, activation='relu'))
        self.model.add(LSTM(100,return_sequences=True, activation='relu'))
        self.model.add(Dense(1, activation="sigmoid"))

        return self.model