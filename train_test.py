import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
import numpy as np
import json

from preprocessing import  get_dataset,prescaler
from RNN_model import RNN
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using the device: {device}")

data_path='/kaggle/input/trading-data/data'
X,Y = get_dataset(data_path)
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float().unsqueeze(1)


#Split Training/Test
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.3, random_state=42) #Tune the split size

#Preprocessing
X_train, X_val_test_scaled = prescaler(X_train, X_val_test, scaler='MinMax') # Possible tuning 'MinMax' or 'Std'

X_val, X_test, Y_val, Y_test = train_test_split(X_val_test_scaled, Y_val_test, test_size=0.5, random_state=42)

X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
X_test = torch.from_numpy(X_test).float()

#Create tensor dataset
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

batch_size=32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


model = RNN(input_size=36,
            hidden_size=256,
            num_layers=8, #Tune the num_layers of GRU
            include_attention=False).to(device)            #FIX THE ATTENTION MODULE AND PUT include_attention=True

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01) #TUNE THIS
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


num_epochs = 20
train_losses=[]
val_losses=[]

#Train loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    print(f'Epoch [{epoch+1}/{num_epochs}]')

    for batch in train_loader:
        optimizer.zero_grad()
        train_data = batch[0].to(device)
        labels = batch[1].to(device)

        output = model(train_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Average Train Loss: {avg_train_loss}")

    #validation
    model.eval()
    val_loss = 0
    correct_predictions = 0
    all_targets=[]
    all_prediction=[]
    with torch.no_grad():
        for val_batch in val_loader:
            val_data = val_batch[0].to(device)
            val_labels = val_batch[1].to(device)


            output = model(val_data)
            val_loss += criterion(output, val_labels).item()

            predicted_labels = (output > 0.5).float()
            correct_predictions += (predicted_labels == val_labels).sum().item()

            all_targets.append(val_labels.cpu().numpy())
            all_prediction.append(predicted_labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = correct_predictions / len(Y_val)
    print(f"Validation Loss: {avg_val_loss} | Validation accuracy: {val_accuracy}")

    all_targets = np.concatenate(all_targets)
    all_prediction = np.concatenate(all_prediction)

    #compute MAE e MSE
    mae = mean_absolute_error(all_targets, all_prediction)  
    mse = mean_squared_error(all_targets, all_prediction)  

    print(f"Validation MAE: {mae} | Validation MSE: {mse}")



# Plot training and validation losses
plot_losses(train_losses, val_losses)

# Testing
model.eval()
test_targets = []
test_predictions = []
with torch.no_grad():
    for test_batch in test_loader:
        test_data, test_labels = test_batch
        test_data, test_labels = test_data.to(device), test_labels.to(device)

        output = model(test_data)
        predicted_labels = (output > 0.5).float()

        test_targets.append(test_labels.cpu().numpy())
        test_predictions.append(predicted_labels.cpu().numpy())

test_targets = np.concatenate(test_targets)
test_predictions = np.concatenate(test_predictions)

# Compute and print classification report and confusion matrix
print("Classification Report:")
print(classification_report(test_targets, test_predictions))

print("Confusion Matrix:")
print(confusion_matrix(test_targets, test_predictions))



torch.save(model.state_dict(), '/kaggle/working/model_weights.pth')

# Save losses and epochs information
training_info = {
    'num_epochs': num_epochs,
    'train_losses': train_losses,
    'val_losses': val_losses
}

with open('/kaggle/working/training_info.json', 'w') as f:
    json.dump(training_info, f)

