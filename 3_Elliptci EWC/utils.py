import torch
from torch import nn
import torch.utils.data
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def variable(t: torch.Tensor, use_cuda=False, **kwargs):
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, old_tasks_loader: NeighborLoader, criterion, total_ele):

        self.model = model
        self.old_tasks_loader = old_tasks_loader
        self.total_ele = total_ele

        # n -> is named parameter
        # p -> is parameter tensor itself
        # basically i create a dictionary of name:parameter_weight_values
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        # creating FIM
        self._precision_matrices = self._diag_fisher(criterion)   # model params for previous task A

        # what's happening here?
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)   # current model params for task B

    def _diag_fisher(self, criterion):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            # setting each parameter to 0 for bow and storing all paramas in precision_matrices
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for batch in self.old_tasks_loader:
            batch = batch.to(device)
            self.model.zero_grad()
            out = self.model(batch)  # Perform a single forward pass.
            out = out.reshape((batch.x.shape[0]))
            y_act = batch.y[:batch.batch_size]
            out = out[:batch.batch_size]
            loss = criterion(out, y_act)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            
            # now for each parameter in the mode i store in the precision_matrices the square of the graident -> FIM formula
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / self.total_ele

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # remember precision_matrices is the gradient and p - self._means([n]) is the difference of the cuurent weight to the old weight
            # this is the second part of eqaution no 9
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader):
    model.train()
    total_loss = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()  # Clear gradients.
        
        out = model(batch)  # Perform a single forward pass.
        out = out.reshape((batch.x.shape[0]))
        
        y_act = batch.y[:batch.batch_size]
        out = out[:batch.batch_size]

        loss = criterion(out, y_act)

        total += y_act.size(0)
        total_loss += loss.item() * y_act.size(0)   # always use loss.item() * y.size(0)
        
        loss.backward()        
        optimizer.step()  # Update parameters based on gradients
        
    average_loss = total_loss / total
    
    return total_loss / total


def ewc_train(model: nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    total_loss = 0
    total = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()  # Clear gradients.
        
        out = model(batch)  # Perform a single forward pass.
        out = out.reshape((batch.x.shape[0]))
        
        y_act = batch.y[:batch.batch_size]
        out = out[:batch.batch_size]
        
        loss = criterion(out, y_act)  + importance * ewc.penalty(model)
        total_loss += loss.item()
        total += y_act.size(0)
        
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients
        
    return total_loss / total

# def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
#     model.eval()
#     correct = 0
#     for input, target in data_loader:
#         input, target = variable(input), variable(target)
#         output = model(input)
#         correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
#     return correct / len(data_loader.dataset)
