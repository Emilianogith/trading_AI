import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking,LSTM, Dense
from sklearn.svm import SVC


print('libraries imported')

def split_dataset(array):
    X=array[:,:-1]
    Y=array[:,-1]

    return X,Y
    
directory='/Users/emilianoparadiso/Desktop/progetto trading/'


dataset_list = []
dataset_list_no_labels = []
labels_list = []

for filename in os.listdir(directory):
    if filename.endswith('.npy'):
        file_path = os.path.join(directory, filename)
        array = np.load(file_path)
        print(len(array))

        dataset_list.append(array)
        
        data_no_labels = array[:, :-1]  
        labels = array[:, -1]  
        
        dataset_list_no_labels.append(data_no_labels)
        labels_list.append(labels)

dataset = np.vstack(dataset_list)

padded_data = pad_sequences(dataset_list_no_labels, dtype='float32', padding='post')
padded_labels = pad_sequences(labels_list, dtype='int32', padding='post')

dataset_RNN = np.array(padded_data)
labels_RNN = np.array(padded_labels)


print("Dataset for linear models:", dataset.shape)
print("Data type:", dataset.dtype)
print("First 5 elements of Dataset:", dataset[:5])
print("Dataset for RNN:", dataset_RNN.shape)
print("Labels for RNN:", labels_RNN.shape)

print("First 5 elements of Dataset:", dataset_RNN[0][:5], labels_RNN[0][:5])


#splitting Dataset
X,Y= split_dataset(dataset)

print("size X:", X.shape)
print("size Y:", Y.shape)

#print("X:", X)
print("Y:", Y)


positive_examples=np.sum(Y==1)
print("positive examples:",positive_examples)
print("negative examples:",len(Y)-positive_examples)


## Preprocessing
abled = True

if abled == True:
    scaler= MinMaxScaler()
    X= scaler.fit_transform(X)

print(X)

#Split Training/Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)



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

    
models = Models(n_timesteps=dataset_RNN.shape[1],n_features=dataset_RNN.shape[2])
rf = models.RandomForest()
svm = models.SVM()
RNN = models.RNN()

#Training
rf.fit(X_train, Y_train)
svm.fit(X_train, Y_train)

epochs = 50
RNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
RNN.summary()

test_block_index = -1

X_RNN_test = dataset_RNN[test_block_index]
X_RNN_test = np.expand_dims(X_RNN_test, axis=0)
Y_RNN_test = labels_RNN[test_block_index]
Y_RNN_test = np.expand_dims(Y_RNN_test, axis=0)

X_RNN_train = np.delete(dataset_RNN, test_block_index, axis=0)
Y_RNN_train = np.delete(labels_RNN, test_block_index, axis=0)

print("X_RNN_test_shape", X_RNN_test.shape)
print("Y_RNN_test_shape", Y_RNN_test.shape)
print("X_RNN_train_shape", X_RNN_train.shape)
print("Y_RNN_train_shape", Y_RNN_train.shape)

RNN.fit(X_RNN_train, Y_RNN_train, epochs=epochs, batch_size=1, verbose=1)

#Evaluation
rf_pred = rf.predict(X_test)
rf_classification_report = classification_report(Y_test, rf_pred)
rf_confusion_matrix = confusion_matrix(Y_test, rf_pred)
print("RandomForest classification report: \n", rf_classification_report)
print("RandomForest confusion matrix: \n", rf_confusion_matrix)


svm_pred = svm.predict(X_test)
svm_classification_report = classification_report(Y_test, svm_pred)
svm_confusion_matrix = confusion_matrix(Y_test, svm_pred)
print("SVM classification report: \n", svm_classification_report)
print("SVM confusion matrix: \n", svm_confusion_matrix)


RNN_pred = RNN.predict(X_RNN_test)
RNN_pred = RNN_pred.squeeze(axis=0) 

timestep_index = 10
predictions_for_timestep = RNN_pred[timestep_index,:]  # Estrai le predizioni per il timestep 10

# Mostra alcune predizioni per il timestep specificato
print("Predizioni per il timestep {}:".format(timestep_index))
print(predictions_for_timestep)

print("RNN_pred size:", RNN_pred.shape)
print( RNN_pred)


RNN_pred_binary = (RNN_pred > 0.5).astype(int)
Y_RNN_test_binary = Y_RNN_test.squeeze(axis=0)

RNN_classification_report = classification_report(Y_RNN_test_binary, RNN_pred_binary)
RNN_confusion_matrix = confusion_matrix(Y_RNN_test_binary, RNN_pred_binary)
print("RNN classification report:\n", RNN_classification_report)
print("RNN confusion matrix:\n", RNN_confusion_matrix)

loss_RNN,accuracy_RNN = RNN.evaluate(X_RNN_test, Y_RNN_test)
print("loss:", loss_RNN)
print("accuracy:", accuracy_RNN)


#K-Fold cross validation 
# KF = KFold(n_splits=5, shuffle=True, random_state= 42)
# scores_rf = cross_val_score(rf,X,Y, cv=KF)
# print("scores Random Forest with K-Fold cross validation",scores_rf)

# scores_svm = cross_val_score(svm,X,Y, cv=KF)
# print("scores SVM with K-Fold cross validation",scores_svm)
