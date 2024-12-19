from typing import *
import torch
import torch.nn as nn

class FidelityEvaluationLoss(nn.Module):
    def __init__(self, num_classes, r, measurement="binary_accuracy", device="cpu"):
        super(FidelityEvaluationLoss, self).__init__()

        measure_funcs = {
            "binary_accuracy": self.calculate_binary_accuracy,
            "class_accuracy": self.calculate_class_accuracy,
        }
        if measurement not in measure_funcs.keys(): 
            raise ValueError("measurement parameter must be either 'binary_accuracy', or 'class_accuracy'")

        self.num_classes = num_classes
        self.r = r
        self.measurement_func = measure_funcs[measurement]
        self.register_buffer("fidelity_costs", torch.tensor([0, r]))
        self.to(device)

    def forward(self, fe_output, lf_output, hf_output, target):
        # start = time.time()
        fe_output = nn.functional.softmax(fe_output, dim=1)
        # print("1", time.time()-start)
        lf_accuracy = self.measurement_func(lf_output.detach(), target)
        hf_accuracy = self.measurement_func(hf_output.detach(), target)
        # print("2", time.time()-start)
        accuracies = torch.stack([lf_accuracy, hf_accuracy], dim=1)
        accuracies = accuracies.to(fe_output.device)
        # print("3", time.time()-start)
        expected_accuracy = torch.sum(fe_output * accuracies, dim=1)
        # print("4", time.time()-start)
        self.fidelity_costs = self.fidelity_costs.to(fe_output.device)
        fidelity_cost = torch.sum(fe_output * self.fidelity_costs, dim=1)
        # print("5", time.time()-start)
        loss = torch.sum(fidelity_cost + (1 - expected_accuracy))
        # print("6", time.time()-start)
        return loss

    def calculate_binary_accuracy(self, pred, target):
        pred_logit = torch.argmax(pred, dim=1)
        try:
            target_logit = torch.argmax(target, dim=1)
        except:
            target_logit = target

        return pred_logit == target_logit
    
    def calculate_class_accuracy(self, pred, target):
        target_logit = torch.argmax(target, dim=1)

        return pred[target_logit]
