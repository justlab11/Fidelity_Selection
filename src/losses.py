from typing import *
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MetaLossFunction(nn.Module):
    def __init__(self, ch: List[float], cw: float, device: str, loss_fun: str="CE", class_weighted: bool=False):
        '''
        Loss function for the FE model
        Parameters:
            ch - cost of using high fidelity
            cw - cost of being wrong
            loss_fun - the choice of loss function used to determine wrong predictions
            class_weighted - upweight samples where fidelities disagree (one correct,
                one wrong) to match the total weight of samples where they agree,
                per batch. Without this, a task where only a small minority of
                samples actually carry routing signal (e.g. LLVIP/YOLO, where LF and
                HF agree on ~94% of samples) lets a degenerate "always pick one
                fidelity" policy minimize expected cost almost for free, since the
                loss is dominated by the indifferent majority. Only meaningful when
                each y_preds tensor is per-sample classification-shaped
                ((batch_size, num_classes) with argmax comparable to y_true) - leave
                off for segmentation-shaped predictions.
        '''
        super().__init__()
        self.ch = ch
        self.ch.insert(0, 0) # first element is zero because no cost for LF sample usage

        self.cw = cw
        self.device = device
        self.class_weighted = class_weighted

        match loss_fun:
            case "CE":
                self.loss_fun = nn.CrossEntropyLoss(reduction="none")
            case "binary":
                self.loss_fun = nn.L1Loss(reduction="none")
            case default:
                raise ValueError("Invalid loss function")

    def forward(self, y_true, y_preds, choices):
        # y_preds: List of tensors, length n_f, each (batch_size, num_classes)
        # choices: (batch_size, n_f)  # FE selection probabilities
        choices = nn.Softmax(dim=1)(choices)
        choices = choices.to(self.device)

        # Compute per-fidelity losses for each sample, stack as (batch_size, n_f)
        model_losses = []
        for i, pred in enumerate(y_preds):
            loss = self.loss_fun(pred, y_true)  # shape: (batch_size,)
            model_losses.append(loss)

        model_losses_tensor = torch.stack(model_losses, dim=1)  # (batch_size, n_f)
        model_losses_tensor = model_losses_tensor.to(self.device)

        # Cost ratio vector; shape (n_f,)
        ch_vec = torch.tensor(self.ch, device=self.device).float()

        # For each sample, total cost: cost ratio + loss
        total_costs = model_losses_tensor + ch_vec  # broadcasting: (batch_size, n_f)

        # For each sample, expected cost under FE probabilities:
        expected_costs = torch.sum(choices * total_costs, dim=1)  # (batch_size,)

        if not self.class_weighted:
            return expected_costs.mean()

        # Per-fidelity correctness, derived the same way compute_gate_routing_details
        # does (argmax(pred) == y_true). "Disagreement" samples (fidelities differ)
        # are the only ones with any routing signal at all; "agreement" samples
        # contribute the same expected cost regardless of the gate's choice modulo
        # the ch penalty. Reweight disagreement samples so their total contribution
        # to the batch loss matches the agreement samples' - i.e. per-batch inverse-
        # frequency class balancing - rather than letting them get drowned out.
        with torch.no_grad():
            fidelity_correct = torch.stack(
                [torch.argmax(pred, dim=1) == y_true for pred in y_preds], dim=1
            )  # (batch_size, n_f)
            disagreement = fidelity_correct.any(dim=1) & ~fidelity_correct.all(dim=1)

            num_disagree = disagreement.sum()
            num_agree = disagreement.numel() - num_disagree

            weights = torch.ones_like(expected_costs)
            if num_disagree > 0 and num_agree > 0:
                weights[disagreement] = num_agree.float() / num_disagree.float()

        return (weights * expected_costs).sum() / weights.sum()
    
    # def forward(self, y_true: torch.tensor, y_preds: torch.tensor, choices: torch.tensor):
    #     batch_size = len(y_true)
    #     model_losses = []
    #     choices = nn.Softmax(dim=1)(choices)
    #     choices = choices.to(self.device)

    #     with torch.no_grad():
    #         for pred in y_preds:
    #             loss = self.loss_fun(pred, y_true)
    #             model_losses.append(loss)

    #     model_losses_tensor = torch.stack(model_losses)

    #     cw_matrix = self.cw * torch.ones_like(model_losses_tensor)

    #     ch_matrix = torch.tensor(self.ch)
    #     ch_matrix = ch_matrix.view(-1, len(self.ch))
    #     ch_matrix = ch_matrix.repeat(batch_size, 1).T

    #     cw_matrix = cw_matrix.to(self.device)
    #     ch_matrix = ch_matrix.to(self.device)
    #     model_losses_tensor = model_losses_tensor.to(self.device)

    #     model_loss_weights = model_losses_tensor * (cw_matrix + ch_matrix)
    #     meta_loss = choices.T * model_loss_weights

    #     loss_result = torch.sum(meta_loss) / batch_size

    #     return loss_result