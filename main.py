import argparse
from simplemodel import SimpleModel
import torch

from utils import *

def evaluate():
    #load from checkpoint
    checkpoint = "./lightning_logs/my_model/version_0/checkpoints/best-checkpoint.ckpt"
    #model = SimpleModel.load_from_checkpoint(checkpoint)
    model = SimpleModel()
    model.model_test(checkpoint)

def inference():
    #load from checkpoint
    checkpoint = "./lightning_logs/my_model/version_0/checkpoints/best-checkpoint.ckpt"
    model = SimpleModel.load_from_checkpoint(checkpoint)

    # get a random sample
    data_path='./data'
    X,Y = get_dataset(data_path)
    indices = torch.randint(0, X.shape[0],(1,))
    x = X[indices]

    inference = model.inference(x)
    print('the correspondent predicted label is: ', inference)
    
def train():
    model = SimpleModel()
    model.model_training()
    

def plot_graphics():
    csv_data = "./lightning_logs/my_model/version_0/metrics.csv"
    plot_train_loss(csv_data)
  

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-i', '--inference', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()
    if args.evaluate:
        evaluate()
    if args.plot:
        plot_graphics()
    if args.inference:
        inference()

    
if __name__ == '__main__':
    main()