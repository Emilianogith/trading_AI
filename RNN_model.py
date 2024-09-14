import torch
import torch.nn as nn

"""
This code contains the model used.
Future works: include biderctionality in GRU.


TO DO: FIX THE ATTENTION MODULE!!!
"""

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        # Definisci un livello lineare per calcolare i punteggi di attenzione
        self.attn_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, gru_outputs, h):
        # gru_outputs: (batch_size, seq_len, hidden_size)
        # h: (num_layers, batch_size, hidden_size) -> consideriamo solo l'ultimo livello h[-1]

        # Estraiamo l'ultimo hidden state (h_final)
        h_final = h[-1].unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Calcoliamo gli attention scores per ogni timestep usando gli hidden states e h_final
        attn_scores = self.attn_weights(gru_outputs + h_final)  # (batch_size, seq_len, 1)
        
        # Applichiamo softmax per normalizzare i punteggi in pesi di attenzione
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)

        # Calcoliamo il contesto come somma pesata degli hidden states
        context = torch.sum(attn_weights * gru_outputs, dim=1)  # (batch_size, hidden_size)
        
        return context




class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, include_attention=True, output_dim = 1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.include_attention = include_attention

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_dim)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(0.5)

        self.attention = AttentionModule(hidden_size)

    
    def forward(self, x):

        x,h = self.gru(x)

        if self.include_attention == True:
            x = self.attention(x, h)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        output= torch.sigmoid(self.dropout(x))

        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':

    print('''Show an example with:
input_size = 36  
hidden_size = 256  
num_layers = 8  
batch_size = 800  
          ''')
    # Example for input size
    input_size = 36  # Number of features
    hidden_size = 256  # GRU hideen units
    num_layers = 8  # Number of GRU layers
    batch_size = 800  # Batch dimension

    # Inizializzazione del modello
    model = RNN(input_size, hidden_size, num_layers, include_attention=True)

    print(f"Total parameters of the model: {count_parameters(model)}")

