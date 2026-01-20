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
        
        # Image-level selection probabilities (matching your routing logic)
        selection_prob_per_image = selection_prob.mean(dim=(1, 2))  # (B,)
        
        # Weight CE by image-level selection
        weighted_ce = selection_prob * ce
        ce_loss = weighted_ce.mean()
        
        # Coverage now measures % of images selected, not pixels
        coverage = selection_prob_per_image.mean()
        
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

    def one_run(self, model, dataloader, hf_model, threshold=0.5, train_body=True, optimizer=None):
        is_training = bool(optimizer)
        model.train(mode=is_training)
        device = next(model.parameters()).device
        hf_model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0  # For classification: samples; for segmentation: pixels
        total_selected_images = 0  # Images selected by SelectiveNet
        total_coverage = 0.0
        
        is_segmentation = None
        
        context_manager = torch.no_grad() if not is_training else torch.enable_grad()
        
        with context_manager:
            for lf_data, hf_data, target in dataloader:
                hf_data = hf_data.to(device, torch.float)
                lf_data = lf_data.to(device, torch.float)
                target = target.type(torch.LongTensor).to(device)
                
                if train_body:
                    selnet_layers = model.selective_forward(lf_data)
                else:
                    selnet_layers = model.selective_head(lf_data)
                
                selnet_output = selnet_layers["output"]
                selnet_aux = selnet_layers["aux"]
                selnet_select = selnet_layers["select"]
                
                if is_segmentation is None:
                    is_segmentation = (selnet_output.dim() == 4)
                
                if optimizer:
                    selnet_loss = self.selective_loss(target, selnet_select, selnet_output, c=self.c)
                    aux_loss = self.aux_ce_loss(target, selnet_aux)
                    loss = self.alpha * selnet_loss + (1-self.alpha) * aux_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * lf_data.size(0)
                
                # === CLASSIFICATION ===
                if not is_segmentation:
                    select_prob = torch.sigmoid(selnet_select).squeeze(1)  # [B]
                    cascade_mask = (select_prob >= threshold)  # [B]
                    coverage = select_prob.mean().item()
                    total_coverage += coverage
                    
                    # SelectiveNet predictions
                    sel_pred = torch.argmax(selnet_output, dim=1)  # [B]
                    sel_correct = (sel_pred[cascade_mask] == target[cascade_mask]).sum().item()
                    
                    # HF model on abstained samples
                    hf_correct = 0
                    abstained_mask = ~cascade_mask
                    if hf_model is not None and abstained_mask.sum() > 0:
                        abstained_data = torch.cat([lf_data[abstained_mask], hf_data[abstained_mask]], dim=1)
                        hf_layers = hf_model(abstained_data)
                        hf_pred = torch.argmax(hf_layers["output"], dim=1)
                        hf_correct = (hf_pred == target[abstained_mask]).sum().item()
                    
                    total_correct += sel_correct + hf_correct
                    total_samples += lf_data.size(0)
                    total_selected_images += cascade_mask.sum().item()
                
                # === SEGMENTATION ===
                else:
                    select_prob_per_pixel = torch.sigmoid(selnet_select).squeeze(1)  # [B, H, W]
                    
                    # Average selection probability per image
                    select_prob_per_image = select_prob_per_pixel.mean(dim=(1, 2))  # [B]
                    
                    # Image-level decision: use SelectiveNet if avg prob >= threshold
                    use_selnet_mask = (select_prob_per_image >= threshold)  # [B]
                    
                    # Coverage is average of pixel-level selection probabilities
                    coverage = select_prob_per_pixel.mean().item()
                    total_coverage += coverage
                    
                    # SelectiveNet predictions on selected images
                    sel_pred = torch.argmax(selnet_output, dim=1)  # [B, H, W]
                    sel_correct = 0
                    for img_idx in torch.where(use_selnet_mask)[0]:
                        sel_correct += (sel_pred[img_idx] == target[img_idx]).sum().item()
                    
                    # HF model on abstained images
                    hf_correct = 0
                    abstained_mask = ~use_selnet_mask  # [B]
                    
                    if hf_model is not None and abstained_mask.sum() > 0:
                        # Run HF model on entire abstained images
                        combined_data = torch.cat([lf_data[abstained_mask], hf_data[abstained_mask]], dim=1)
                        hf_layers = hf_model(combined_data)
                        hf_output = hf_layers["output"]  # [num_abstained, C, H, W]
                        hf_pred = torch.argmax(hf_output, dim=1)  # [num_abstained, H, W]
                        
                        # Count correct pixels on abstained images
                        for batch_idx, img_idx in enumerate(torch.where(abstained_mask)[0]):
                            hf_correct += (hf_pred[batch_idx] == target[img_idx]).sum().item()
                    
                    num_pixels = target.numel()
                    total_correct += sel_correct + hf_correct
                    total_samples += num_pixels
                    total_selected_images += use_selnet_mask.sum().item()
        
        avg_loss = total_loss / max(len(dataloader) * dataloader.batch_size, 1)
        avg_coverage = total_coverage / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # For segmentation: this is % of images routed to SelectiveNet
        # For classification: this is % of samples routed to SelectiveNet
        selection_rate = total_selected_images / (len(dataloader) * dataloader.batch_size)
        usage = 1 - selection_rate
        
        return {
            'loss': avg_loss, 
            'accuracy': accuracy, 
            'coverage': avg_coverage,  # Average pixel-level selection probability
            'selection_rate': selection_rate,  # % of images using SelectiveNet
            'usage': usage
        }