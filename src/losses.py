from typing import *
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MetaLossFunction(nn.Module):
    def __init__(self, ch: List[float], cw: float, device: str, loss_fun: str="CE"):
        '''
        Loss function for the FE model.

        Each fidelity's per-sample loss against the true label is computed
        once, upstream (see helpers.compute_fidelity_loss_correct,
        save_latent) - this function only ever combines those already-computed
        numbers with the gate's own routing probabilities, so it no longer
        needs to know anything about a fidelity's output shape (classification
        logits vs. segmentation's per-pixel map vs. YOLO's detections all look
        the same by the time they get here).

        Parameters:
            ch - cost of using high fidelity
            cw - cost of being wrong
            loss_fun - retained for interface/config parity; no longer used
                directly here, since fidelity losses arrive precomputed - see
                compute_fidelity_loss_correct for what actually produces them.
        '''
        super().__init__()
        self.ch = ch
        self.ch.insert(0, 0) # first element is zero because no cost for LF sample usage

        self.cw = cw
        self.device = device
        self.loss_fun = loss_fun

    def forward(self, losses: List[torch.Tensor], gate_logits: torch.Tensor):
        """
        losses - list of length n_f, each a (batch_size,) precomputed per-sample
            loss for that fidelity.
        gate_logits - (batch_size, n_f), the gate model's own raw routing logits
            (pre-softmax).
        """
        choices = nn.Softmax(dim=1)(gate_logits)
        choices = choices.to(self.device)

        model_losses_tensor = torch.stack([loss.to(self.device) for loss in losses], dim=1)  # (batch_size, n_f)

        # Cost ratio vector; shape (n_f,)
        ch_vec = torch.tensor(self.ch, device=self.device).float()

        # For each sample, total cost: cost ratio + loss (Eq. 3/4 in the paper -
        # additive, not multiplicative; cw only appears in the illustrative 0-1-loss
        # special case, not the general loss). A multiplicative loss*(cw+ch) was
        # tried here but is degenerate: whenever a fidelity's own loss is already
        # near 0 (e.g. HF confidently correct), multiplying by any finite (cw+ch)
        # keeps its cost near 0 regardless of ch, so no ch value can ever price it
        # out - usage asymptotes well above 0% instead of sweeping down to it.
        total_costs = model_losses_tensor + ch_vec  # broadcasting: (batch_size, n_f)

        # For each sample, expected cost under FE probabilities:
        expected_costs = torch.sum(choices * total_costs, dim=1)  # (batch_size,)

        return expected_costs.mean()
    
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