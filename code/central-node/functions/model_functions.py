from flask import current_app
import torch
import torch.nn as nn

from torch.optim import SGD

from functions.data_functions import *

class FederatedLogisticRegression(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1, bias=bias)
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        return self.linear(x).view(-1)

    @staticmethod
    def train_step(model, batch):
        x, y = batch
        out = model(x)
        loss = model.loss(out, y)
        return loss

    @staticmethod
    def test_step(model, batch):
        x, y = batch
        out = model(x)
        loss = model.loss(out, y)
        preds = out > 0 # Predict y = 1 if P(y = 1) > 0.5
        corrects = torch.tensor(torch.sum(preds == y).item())
        return loss, corrects

    @staticmethod
    def get_parameters(model):
        return model.state_dict()

    @staticmethod
    def apply_parameters(model, parameters):
        model.load_state_dict(parameters)

def get_loaders() -> any:
    GLOBAL_SEED = current_app.config['GLOBAL_SEED']
    GLOBAL_SAMPLE_RATE = current_app.config['GLOBAL_SAMPLE_RATE']
   
    train_tensor = torch.load('tensors/train.pt')
    test_tensor = torch.load('tensors/test.pt')

    train_loader = DataLoader(
        train_tensor,
        batch_size = int(len(train_tensor) * GLOBAL_SAMPLE_RATE),
        generator = torch.Generator().manual_seed(GLOBAL_SEED)
    )
    test_loader = DataLoader(test_tensor, 64)
    return train_loader,test_loader

def train(
    model: any,
    train_loader: any
):
    GLOBAL_MODEL_OPTIMIZER = current_app.config['GLOBAL_MODEL_OPTIMIZER']
    GLOBAL_LEARNING_RATE = current_app.config['GLOBAL_LEARNING_RATE']    
    GLOBAL_TRAINING_EPOCHS = current_app.config['GLOBAL_TRAINING_EPOCHS']

    opt_func = None
    if GLOBAL_MODEL_OPTIMIZER == 'SGD':
        opt_func = torch.optim.SGD

    optimizer = opt_func(model.parameters(), GLOBAL_LEARNING_RATE)
    model_type = type(model)
    
    for epoch in range(GLOBAL_TRAINING_EPOCHS):
        losses = []
        for batch in train_loader:
            loss = model_type.train_step(model, batch)
            loss.backward()
            losses.append(loss)
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch {}, loss = {}".format(epoch + 1, torch.sum(loss) / len(train_loader)))

def test(
    model: any, 
    test_loader: any
) -> any:
    with torch.no_grad():
        losses = []
        accuracies = []
        total_size = 0
        
        for batch in test_loader:
            total_size += len(batch[1])
            loss, corrects = model.test_step(model, batch)
            losses.append(loss)
            accuracies.append(corrects)

        average_loss = np.array(loss).sum() / total_size
        total_accuracy = np.array(accuracies).sum() / total_size
        return average_loss, total_accuracy

def initial_model_training():
    print('Initial model training')
    GLOBAL_SEED = current_app.config['GLOBAL_SEED']
    GLOBAL_INPUT_SIZE = current_app.config['INPUT_SIZE']

    torch.manual_seed(GLOBAL_SEED)
    #print('Loaders')
    given_train_loader, given_test_loader = get_loaders()

    lr_model = FederatedLogisticRegression(dim = GLOBAL_INPUT_SIZE)
    #print('Train')
    train(
        model = lr_model, 
        train_loader = given_train_loader,  
    )
    #print('Test')
    average_loss, total_accuracy = test(
        model = lr_model, 
        test_loader = given_test_loader
    )
    print('Loss:',average_loss)
    print('Accuracy:',total_accuracy)
    parameters = lr_model.get_parameters(lr_model)
    torch.save(parameters, 'models/initial_model_parameters.pth')

