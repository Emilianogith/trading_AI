import numpy
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from utils import *


class SimpleModel(LightningModule):
    BATCH_SIZE = 32
    NUM_EPOCHS =300

    def __init__(self, input_dim = 36, hidden_dim_1=126, hidden_dim_2 = 126, output_dim = 1,  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        # Architecture
        self.dense1 = nn.Linear(input_dim, hidden_dim_1, dtype =torch.float64)
        self.dense2 = nn.Linear(hidden_dim_1, hidden_dim_2, dtype =torch.float64)
        self.dense3 = nn.Linear(hidden_dim_2, output_dim, dtype =torch.float64)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        self.loss_fn = nn.BCELoss()

        self.test_preds = []
        self.test_targets = []

        # Hyperparameters to save 
        self.save_hyperparameters({
            "input_dim": input_dim,
            "hidden_dim_1": hidden_dim_1,
            "hidden_dim_2": hidden_dim_2,
            "batch_size": self.BATCH_SIZE,
            "dropout": 0.5,
            "lr": 1e-3
        })

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.relu(self.dense2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.dense3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(1)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        self.eval()
        with torch.no_grad():
            y_hat = self(x)
        y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat, y)

        preds = (y_hat > 0.5).int()
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        return loss
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)
        cm = confusion_matrix(targets, preds)

        print(f"\nTest Accuracy:  {acc:.4f}")
        print(f"Test F1 Score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def model_training(self):

        #Preporcessing:
        ...
        
        train_dataset = torch.load('./train_dataset.pt', map_location='cpu', weights_only=False)
        val_dataset = torch.load('./val_dataset.pt', map_location='cpu', weights_only=False)

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True) 
        val_loader   = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, num_workers=4, persistent_workers=True)

        print(
            f"Info during training:\n"
            f"train_len: {len(train_dataset)}\n"
            f"val_len: {len(val_dataset)}\n"
            f"test_len: {len(val_dataset)}\n"
        )

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",            # metric to monitor
            mode="min",                    # "min" menas minimize the metric
            save_top_k=1,                  
            filename="best-checkpoint",   
            verbose=True
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,     # minimum change to qualify as improvement
            patience=10,         # number of epochs with no improvement to wait
            verbose=True,
            mode="min"
        )

        csv_logger = CSVLogger("lightning_logs", name="my_model")

        trainer = Trainer(
            max_epochs=self.NUM_EPOCHS,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger = csv_logger,
            log_every_n_steps=32
        )

        trainer.fit(self, train_loader, val_loader)
    
    def model_test(self, best_checkpoint):
        test_dataset = torch.load('./test_dataset.pt', map_location='cpu', weights_only=False)
        test_loader  = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, num_workers=4, persistent_workers=True)

        trainer = Trainer(logger=False)
        best_model = SimpleModel.load_from_checkpoint(best_checkpoint)

        # Evaluate on test set
        trainer.test(best_model, dataloaders=test_loader)
        

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)

        output = 0
        if logits > 0.5:
            output =1
        return output
        
    def move_to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret