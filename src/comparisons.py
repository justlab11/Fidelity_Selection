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
    def __init__(self, model_folder, alpha=.7):
        self.model_folder = model_folder
        self.alpha = alpha

    def selective_loss_class(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        selection_prob = torch.sigmoid(selection_logits)

        ce = F.cross_entropy(lf_logits, y_true.argmax(dim=1), reduction='none')
        weighted_ce = selection_prob.view(-1) * ce
        ce_loss = weighted_ce.mean()

        coverage = selection_prob.mean()
        penalty = lamda * torch.clamp(-coverage + c, min=0) ** 2

        return ce_loss + penalty

    def selective_loss_seg(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        # selection per pixel: (B, 1, H, W) -> (B, H, W)
        selection_prob = torch.sigmoid(selection_logits).squeeze(1)

        # per-pixel CE: (B, H, W)
        ce = F.cross_entropy(lf_logits, y_true, reduction='none')

        # gate by selection prob, then normalize by number of selected pixels
        # (to avoid the model minimizing loss by selecting almost nothing)
        weighted_ce = selection_prob * ce
        # add a small eps to avoid division-by-zero early in training
        eps = 1e-6
        ce_loss = weighted_ce.sum() / (selection_prob.sum() + eps)

        # coverage is now fraction of selected pixels
        coverage = selection_prob.mean()

        # same coverage penalty as in classification
        penalty = lamda * torch.clamp(-coverage + c, min=0) ** 2

        return ce_loss + penalty
    
    def selective_loss(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        if lf_logits.dim() == 2:  # [B, C]
            return self.selective_loss_class(y_true, selection_logits, lf_logits,
                                            lamda=lamda, c=c)
        elif lf_logits.dim() == 4:  # [B, C, H, W]
            return self.selective_loss_seg(y_true, selection_logits, lf_logits,
                                        lamda=lamda, c=c)
        else:
            raise ValueError(f"Unsupported lf_logits shape: {lf_logits.shape}")

    def aux_ce_loss(self, y_true, aux_logits):
        if aux_logits.dim() == 2:
            # classification: aux_logits [B, C], y_true [B] (class indices)
            return F.cross_entropy(aux_logits, y_true, reduction='mean')
        elif aux_logits.dim() == 4:
            # segmentation: aux_logits [B, C, H, W], y_true [B, H, W]
            return F.cross_entropy(aux_logits, y_true, reduction='mean')
        else:
            raise ValueError(f"Unsupported aux_logits shape: {aux_logits.shape}")

    def one_run(self, model, dataloader, train_body=True, optimizer=None):
        model.train(mode=bool(optimizer))
        device = next(model.parameters()).device
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_high = 0

        for data, _, target in dataloader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device, torch.float), target.to(device)
            
            if train_body:
                layers = model.selective_forward(data)
                output = layers["output"]
                aux = layers["aux"]
                select = layers["select"]

            else:
                layers = model.selective_head(data)
                output = layers["output"]
                aux = layers["aux"]
                select = layers["select"]

            sel_loss = self.selective_loss(
                y_true = target,
                selection_logits = select,
                lf_logits = output
            )
            aux_loss = self.aux_ce_loss(
                y_true = target,
                aux_logits = aux
            )

            loss = self.alpha * sel_loss + (1-self.alpha) * aux_loss

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)


