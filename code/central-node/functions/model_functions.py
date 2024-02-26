from flask import current_app
import torch
import torch.nn as nn 

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from functions.data_functions import * 
from functions.storage_functions import *
# Refactored and works
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
        return loss, preds
    
    @staticmethod
    def prediction(model, x):
        out = model(x)
        preds = out > 0
        return preds

    @staticmethod
    def get_parameters(model):
        return model.state_dict()

    @staticmethod
    def apply_parameters(model, parameters):
        model.load_state_dict(parameters)
# Refactored and works
def get_train_test_loaders(
    global_parameters: any 
) -> any:
    train_tensor = torch.load('tensors/train.pt')
    test_tensor = torch.load('tensors/test.pt')

    train_loader = DataLoader(
        train_tensor,
        batch_size = int(len(train_tensor) * global_parameters['sample-rate']),
        generator = torch.Generator().manual_seed(global_parameters['seed'])
    )
    test_loader = DataLoader(test_tensor, 64)
    return train_loader,test_loader
# Refactored and works
def train(
    model: any,
    train_loader: any,
    global_parameters: any
):
    opt_func = None
    if global_parameters['optimizer'] == 'SGD':
        opt_func = torch.optim.SGD

    optimizer = opt_func(model.parameters(), global_parameters['learning-rate'])
    model_type = type(model)
    
    for epoch in range(global_parameters['epochs']):
        losses = []
        for batch in train_loader:
            loss = model_type.train_step(model, batch)
            loss.backward()
            losses.append(loss)
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch {}, loss = {}".format(epoch + 1, torch.sum(loss) / len(train_loader)))
# Refactored and works
def test(
    model: any, 
    test_loader: any
) -> any:
    with torch.no_grad():
        total_size = 0
        total_confusion_matrix = [0,0,0,0]
        
        for batch in test_loader:
            total_size += len(batch[1])
            _, correct = batch
            _, preds = model.test_step(model, batch)
            
            formated_correct = correct.numpy()
            formated_preds = preds.numpy().astype(int)
            
            tn, fp, fn, tp = confusion_matrix(
                y_true = formated_correct,
                y_pred = formated_preds,
                labels = [0,1]
            ).ravel()

            total_confusion_matrix[0] += int(tp) # True positive
            total_confusion_matrix[1] += int(fp) # False positive
            total_confusion_matrix[2] += int(tn) # True negative
            total_confusion_matrix[3] += int(fn) # False negative
 
        TP, FP, TN, FN = total_confusion_matrix
        # Zero divsion can happen
        TPR = 0
        TNR = 0
        PPV = 0
        FNR = 0
        FPR = 0
        BA = 0
        ACC = 0
        try:
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            PPV = TP/(TP+FP)
            FNR = FN/(FN+TP)
            FPR = FP/(FP+TN)
            BA = (TPR+TNR)/2
            ACC = (TP + TN)/(TP + TN + FP + FN)
        except Exception as e:
            current_app.logger.warning(e)
        
        metrics = {
            'true-positives': TP,
            'false-positives': FP,
            'true-negatives': TN,
            'false-negatives': FN,
            'recall': float(round(TPR,5)),
            'selectivity': float(round(TNR,5)),
            'precision': float(round(PPV,5)),
            'miss-rate': float(round(FNR,5)),
            'fall-out': float(round(FPR,5)),
            'balanced-accuracy': float(round(BA,5)),
            'accuracy': float(round(ACC,5))
        }
        
        return metrics
# Refactored and works
def initial_model_training(
    logger: any,
    global_parameters: any
) -> bool:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    if not training_status['parameters']['start']:
        return False

    if not training_status['parameters']['data-split'] or not training_status['parameters']['preprocessed']:
        return False

    if training_status['parameters']['trained']:
        return False

    model_path = 'models/global_model_0.pth'

    torch.manual_seed(global_parameters['seed'])
    
    given_train_loader, given_test_loader = get_train_test_loaders(
        global_parameters = global_parameters
    )

    lr_model = FederatedLogisticRegression(dim = global_parameters['input-size'])
    
    train(
        model = lr_model, 
        train_loader = given_train_loader,  
        global_parameters = global_parameters
    )
    
    test_metrics = test(
        model = lr_model, 
        test_loader = given_test_loader
    )

    store_global_metrics(
        metrics = test_metrics
    )
    
    parameters = lr_model.get_parameters(lr_model)
    torch.save(parameters, model_path)
    
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)
    training_status['parameters']['trained'] = True
    with open(training_status_path, 'w') as f:
        json.dump(training_status, f, indent=4) 

    return True
# Refactored and works
def model_inference(
    input: any,
    cycle: int
) -> any:
    training_status_path = 'logs/training_status.txt'
    if not os.path.exists(training_status_path):
        return False
    # Might be useful for recoding inference amounts
    training_status = None
    with open(training_status_path, 'r') as f:
        training_status = json.load(f)

    GLOBAL_PARAMETERS = current_app.config['GLOBAL_PARAMETERS']

    global_model_path = 'models/global_model_' + str(cycle) + '.pth'
    if not os.path.exists(global_model_path):
        return None
    
    given_parameters = torch.load(global_model_path)
    
    lr_model = FederatedLogisticRegression(dim = GLOBAL_PARAMETERS['input-size'])
    lr_model.apply_parameters(lr_model, given_parameters)
    
    given_input = torch.tensor(np.array(input, dtype=np.float32))

    with torch.no_grad():
        output = lr_model.prediction(lr_model,given_input)

    return output.tolist()