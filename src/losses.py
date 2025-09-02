from typing import *
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MetaLossFunction(nn.Module):
    def __init__(self, ch: List[float], cw: float, device: str, loss_fun: str="CE"):
        '''
        Loss function for the FE model
        Parameters:
            ch - cost of using high fidelity
            cw - cost of being wrong
            loss_fun - the choice of loss function used to determine wrong predictions
        '''
        super().__init__()
        self.ch = ch
        self.ch.insert(0, 0) # first element is zero because no cost for LF sample usage

        self.cw = cw
        self.device = device

        match loss_fun:
            case "CE":
                self.loss_fun = nn.CrossEntropyLoss(reduction="none")
            case "binary":
                self.loss_fun = nn.L1Loss(reduction="none")
            case default:
                raise ValueError("Invalid loss function")

    def forward(self, y_true: torch.tensor, y_preds: torch.tensor, choices: torch.tensor):
        batch_size = len(y_true)
        model_losses = []
        choices = nn.Softmax(dim=1)(choices)
        choices = choices.to(self.device)

        with torch.no_grad():
            for pred in y_preds:
                loss = self.loss_fun(pred, y_true)
                model_losses.append(loss)

        model_losses_tensor = torch.stack(model_losses)

        cw_matrix = self.cw * torch.ones_like(model_losses_tensor)

        ch_matrix = torch.tensor(self.ch)
        ch_matrix = ch_matrix.view(-1, len(self.ch))
        ch_matrix = ch_matrix.repeat(batch_size, 1).T

        cw_matrix = cw_matrix.to(self.device)
        ch_matrix = ch_matrix.to(self.device)
        model_losses_tensor = model_losses_tensor.to(self.device)

        model_loss_weights = model_losses_tensor * (cw_matrix + ch_matrix)
        meta_loss = choices.T * model_loss_weights

        loss_result = torch.sum(meta_loss) / batch_size

        return loss_result