import torch.backends.cudnn as cudnn
import torch

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from model import ViT


class Solver(object):
    
    def __init__(self, trainloader, testloader, config):
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self):

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Define model and initialize training configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
        
        model = ViT((3, 32, 32), n_patches=8, n_blocks=6, hidden_dim=192, n_heads=8, n_classes=10).to(device)
        
        N_EPOCHS = 5
        LR = 0.005
        optimizer = Adam(model.parameters(), lr=LR)
        criterion = CrossEntropyLoss()

        # Start training
        for epoch in trange(N_EPOCHS, desc="training"):
            train_loss = 0.0
            for batch in tqdm(self.trainloader, desc=f"Epoch {epoch + 1} in training", leave=False):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)

                train_loss += loss.detach().cpu().item() / len(self.trainloader)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        # Test loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(self.testloader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(self.testloader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
            print(f"Test loss: {test_loss:.2f}")
            print(f"Test accuracy: {correct / total * 100:.2f}%")  

