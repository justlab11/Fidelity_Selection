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
    def __init__(self, model_folder, alpha=0.5, c=0.8):
        self.model_folder = model_folder
        self.alpha = alpha
        self.c = c

    def selective_loss_class(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        selection_prob = torch.sigmoid(selection_logits)
        ce = F.cross_entropy(lf_logits, y_true, reduction='none')
        weighted_ce = selection_prob.view(-1) * ce

        ce_loss = weighted_ce.mean()
        coverage = selection_prob.mean()
        
        penalty = lamda * torch.clamp(-coverage + c, min=0) ** 2
        return ce_loss + penalty

    def selective_loss_seg(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        selection_prob = torch.sigmoid(selection_logits).squeeze(1)  # (B, H, W)
        ce = F.cross_entropy(lf_logits, y_true, reduction='none')  # (B, H, W)
        weighted_ce = selection_prob * ce
        
        eps = 1e-6
        ce_loss = weighted_ce.sum() / (selection_prob.sum().detach() + eps)
        coverage = selection_prob.mean()
        penalty = lamda * torch.clamp(-coverage + c, min=0) ** 2
        return ce_loss + penalty
    
    def selective_loss(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        if lf_logits.dim() == 2:  # [B, C]
            return self.selective_loss_class(y_true, selection_logits, lf_logits, lamda=lamda, c=c)
        elif lf_logits.dim() == 4:  # [B, C, H, W]
            return self.selective_loss_seg(y_true, selection_logits, lf_logits, lamda=lamda, c=c)
        else:
            raise ValueError(f"Unsupported lf_logits shape: {lf_logits.shape}")

    def aux_ce_loss(self, y_true, aux_logits):
        if aux_logits.dim() == 2:
            return F.cross_entropy(aux_logits, y_true, reduction='mean')
        elif aux_logits.dim() == 4:
            return F.cross_entropy(aux_logits, y_true, reduction='mean')
        else:
            raise ValueError(f"Unsupported aux_logits shape: {aux_logits.shape}")

    def one_run(self, model, dataloader, hf_model=None, threshold=0.5, train_body=True, optimizer=None):
        is_training = bool(optimizer)
        model.train(mode=is_training)
        device = next(model.parameters()).device
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_high_conf = 0
        total_coverage = 0.0
        
        context_manager = torch.no_grad() if not is_training else torch.enable_grad()
        
        with context_manager:
            for lf_data, hf_data, target in dataloader:
                hf_data = torch.cat([lf_data, hf_data], dim=1)
                hf_data = hf_data.to(device, torch.float)
                lf_data = lf_data.to(device, torch.float)
                target = target.type(torch.LongTensor).to(device)
                
                # SelectiveNet ALWAYS uses LF data
                if train_body:
                    sel_layers = model.selective_forward(lf_data)
                else:
                    sel_layers = model.selective_head(lf_data)
                
                sel_output = sel_layers["output"]
                sel_aux = sel_layers["aux"]
                sel_select = sel_layers["select"]
                
                if optimizer:
                    # Training: selective loss only
                    sel_loss = self.selective_loss(target, sel_select, sel_output, c=self.c)
                    aux_loss = self.aux_ce_loss(target, sel_aux)
                    loss = self.alpha * sel_loss + (1-self.alpha) * aux_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * lf_data.size(0)
                
                # Selection logic
                if sel_select.dim() == 2:  # Classification
                    select_prob = torch.sigmoid(sel_select).squeeze(1)
                else:  # Segmentation
                    select_prob = torch.sigmoid(sel_select).squeeze(1).mean(dim=(1,2))
                
                cascade_mask = (select_prob >= threshold).float()
                coverage = select_prob.mean().item()
                total_coverage += coverage
                
                # SelectiveNet predictions where confident
                sel_pred = torch.argmax(sel_output, dim=1)
                sel_correct = (sel_pred[cascade_mask.bool()] == target[cascade_mask.bool()]).sum().item()
                
                # HF model on abstained (cat LF+HF)
                hf_correct = 0
                if hf_model is not None and (1-cascade_mask).sum() > 0:
                    abstained_mask = (1-cascade_mask).bool()
                    abstained_data = torch.cat([lf_data[abstained_mask], hf_data[abstained_mask]], dim=1)
                    hf_layers = hf_model(abstained_data)
                    hf_pred = torch.argmax(hf_layers["output"], dim=1)
                    hf_correct = (hf_pred == target[abstained_mask]).sum().item()
                
                total_correct += sel_correct + hf_correct
                total_samples += lf_data.size(0)
                total_high_conf += cascade_mask.sum().item()
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_coverage = total_coverage / len(dataloader)
        accuracy = total_correct / total_samples
        high_conf_rate = total_high_conf / total_samples
        
        return {
            'loss': avg_loss, 'accuracy': accuracy, 'coverage': avg_coverage,
            'high_conf_rate': high_conf_rate, 'total_samples': total_samples
        }
