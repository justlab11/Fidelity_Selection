import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SoftmaxResponseMethod:
    def __init__(self, val_dl, test_dl):
        self.val_dl = val_dl
        self.test_dl = test_dl

    def get_softmax_thresholds(self, usage_list):
        logger.info("\nRunning Softmax Response on LF Model")
        
        # covert to what other works use
        coverage_list = [1-i for i in usage_list]

        # collect max probabilities
        all_max_probs = []
        for _, model_output, _, _ in self.val_dl:
            probs = F.softmax(model_output, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            all_max_probs.extend(max_probs.cpu().numpy())

        all_max_probs = np.array(all_max_probs)

        # sort descending for coverage mapping
        sorted_probs = np.sort(all_max_probs)[::-1]
        acc_vals = {}
        usage_vals = {}
        
        for coverage in coverage_list:
            idx = max(0, int(np.floor(len(sorted_probs) * coverage)) - 1)
            threshold = sorted_probs[idx]
            test_acc, test_use = self.apply_softmax_threshold(threshold)

            test_acc *= 100
            test_use *= 100

            logger.info(f"\n\tVal Usage {1-coverage} occurs at threshold {threshold}")
            logger.info(f"\tTest Usage: {test_use:.2f} with accuracy: {test_acc:.2f}")

            acc_vals[threshold] = test_acc
            usage_vals[threshold] = test_use

        return acc_vals, usage_vals

    def apply_softmax_threshold(self, threshold):
        correct = 0
        total = 0
        admitted = 0

        for _, model_output, _, label in self.test_dl:
            probs = F.softmax(model_output, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            for i in range(len(label)):
                total += 1
                if max_probs[i] >= threshold:
                    admitted += 1
                    if preds[i] == label[i]:
                        correct += 1

        accuracy = correct / admitted if admitted > 0 else float('nan')
        coverage = admitted / total if total > 0 else float('nan')
        usage = 1-coverage

        return accuracy, usage

class SelectiveNetMethod:
    def __init__(self, train_dl, val_dl, test_dl, model_folder):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model_folder = model_folder

    

    def selective_loss(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        selection_prob = torch.sigmoid(selection_logits)
        ce = F.cross_entropy(lf_logits, y_true.argmax(dim=1), reduction='none')
        weighted_ce = selection_prob.view(-1) * ce
        ce_loss = weighted_ce.mean()
        coverage = selection_prob.mean()
        penalty = lamda * torch.clamp(-coverage + c, min=0) ** 2
        return ce_loss + penalty
