import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import Dataset

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
    def __init__(self, model_folder, alpha=0.5, c=0.8, lmda=32):
        self.model_folder = model_folder
        self.alpha = alpha
        self.c = c
        self.lmda = lmda

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

    def one_run(self, model, dataloader, hf_model, threshold=0.5, train_body=True, optimizer=None, hf_input_mode: str = "concat"):
        from helpers import assemble_hf_input

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
                    selnet_loss = self.selective_loss(target, selnet_select, selnet_output, c=self.c, lamda=self.lmda)
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
                        abstained_data = assemble_hf_input(lf_data[abstained_mask], hf_data[abstained_mask], hf_input_mode)
                        hf_layers = hf_model(abstained_data) if train_body else hf_model.head(abstained_data)
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
                        combined_data = assemble_hf_input(lf_data[abstained_mask], hf_data[abstained_mask], hf_input_mode)
                        hf_layers = hf_model(combined_data) if train_body else hf_model.head(combined_data)
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

class IndexedDataset(Dataset):
    """Wraps a dataset to also yield each sample's absolute index, so a per-sample
    EMA target buffer can be kept correct under a shuffled DataLoader."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return (idx, *self.base_dataset[idx])

class SelfAdaptiveTrainingMethod:
    """Self-Adaptive Training, selective-classification application (Eq. 12): the
    model gets one extra 'abstain' output class, a per-sample EMA-tracked soft
    target is held across epochs, and inference escalates to the HF model whenever
    the abstain probability exceeds tau (mirroring how SelectiveNet/SR are already
    adapted to this repo's LF/HF cascade convention).
    """
    def __init__(self, num_train_samples, num_classes, alpha=0.99, warmup_epochs=0,
                 is_segmentation=False, device="cpu"):
        self.num_classes = num_classes
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.is_segmentation = is_segmentation
        self.device = device
        self.targets = torch.zeros(num_train_samples, num_classes + 1, device=device)

    def initialize_targets(self, indexed_dataloader):
        with torch.no_grad():
            for idx, _, _, target in indexed_dataloader:
                idx = idx.to(self.device)
                if self.is_segmentation:
                    target = torch.mode(target.reshape(target.size(0), -1), dim=1).values
                target = target.long().to(self.device)

                one_hot = torch.zeros(target.size(0), self.num_classes + 1, device=self.device)
                one_hot.scatter_(1, target.unsqueeze(1), 1.0)
                self.targets[idx] = one_hot

    def update_targets(self, idx, probs, epoch):
        # E_s (warmup_epochs) = 0 for the selective-classification defaults, so
        # updates start from the very first epoch (no warmup).
        if epoch < self.warmup_epochs:
            return
        with torch.no_grad():
            self.targets[idx] = self.alpha * self.targets[idx] + (1 - self.alpha) * probs.detach()

    def sat_loss(self, logits, target_y, idx, eps=1e-8):
        probs = F.softmax(logits, dim=1)
        t_scalar = self.targets[idx, target_y]
        p_true = probs.gather(1, target_y.unsqueeze(1)).squeeze(1)
        p_abstain = probs[:, self.num_classes]

        loss = -(t_scalar * torch.log(p_true + eps) + (1 - t_scalar) * torch.log(p_abstain + eps))
        return loss.mean()

    def one_run(self, model, dataloader, hf_model, tau=0.5, train_body=True, optimizer=None, epoch=0, hf_input_mode: str = "concat"):
        from helpers import assemble_hf_input

        is_training = bool(optimizer)
        model.train(mode=is_training)
        device = next(model.parameters()).device
        hf_model.eval()

        # self.targets (and everything indexed alongside it, e.g. target_y in
        # sat_loss) must live on the same device as the model — sync once here
        # instead of trusting the constructor's device arg to already match.
        if self.targets.device != device:
            self.targets = self.targets.to(device)
        self.device = device

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        total_escalated = 0
        total_units = 0  # images (segmentation) or samples (classification)

        context_manager = torch.no_grad() if not is_training else torch.enable_grad()

        with context_manager:
            for idx, lf_data, hf_data, target in dataloader:
                idx = idx.to(device)
                lf_data = lf_data.to(device, torch.float)
                hf_data = hf_data.to(device, torch.float)
                target = target.long().to(device)

                logits = model(lf_data)["output"] if train_body else model.head(lf_data)["output"]

                if not self.is_segmentation:
                    probs = F.softmax(logits, dim=1)

                    if is_training:
                        self.update_targets(idx, probs, epoch)
                        loss = self.sat_loss(logits, target, idx)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * lf_data.size(0)
                        probs = probs.detach()

                    abstain_prob = probs[:, self.num_classes]
                    escalate_mask = abstain_prob > tau
                    kept_mask = ~escalate_mask

                    own_pred = torch.argmax(logits[:, :self.num_classes], dim=1)
                    correct = (own_pred[kept_mask] == target[kept_mask]).sum().item()

                    if hf_model is not None and escalate_mask.sum() > 0:
                        escalated_data = assemble_hf_input(lf_data[escalate_mask], hf_data[escalate_mask], hf_input_mode)
                        hf_layers = hf_model(escalated_data) if train_body else hf_model.head(escalated_data)
                        hf_pred = torch.argmax(hf_layers["output"], dim=1)
                        correct += (hf_pred == target[escalate_mask]).sum().item()

                    total_correct += correct
                    total_samples += lf_data.size(0)
                    total_escalated += escalate_mask.sum().item()
                    total_units += lf_data.size(0)

                else:
                    # Per-pixel EMA targets are intractable at full resolution, so
                    # SAT tracks/decides at image level here (mean-pooled logits,
                    # majority-vote pixel label) — same granularity SelectiveNet's
                    # own segmentation branch already uses for its bookkeeping.
                    image_logits = logits.mean(dim=(2, 3))
                    majority_label = torch.mode(target.reshape(target.size(0), -1), dim=1).values

                    probs_image = F.softmax(image_logits, dim=1)

                    if is_training:
                        self.update_targets(idx, probs_image, epoch)
                        loss = self.sat_loss(image_logits, majority_label, idx)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * lf_data.size(0)
                        probs_image = probs_image.detach()

                    abstain_prob = probs_image[:, self.num_classes]
                    escalate_mask = abstain_prob > tau
                    kept_mask = ~escalate_mask

                    own_pred = torch.argmax(logits[:, :self.num_classes], dim=1)  # [B, H, W]
                    correct = (own_pred[kept_mask] == target[kept_mask]).sum().item()

                    if hf_model is not None and escalate_mask.sum() > 0:
                        escalated_data = assemble_hf_input(lf_data[escalate_mask], hf_data[escalate_mask], hf_input_mode)
                        hf_layers = hf_model(escalated_data) if train_body else hf_model.head(escalated_data)
                        hf_pred = torch.argmax(hf_layers["output"], dim=1)
                        correct += (hf_pred == target[escalate_mask]).sum().item()

                    total_correct += correct
                    total_samples += target.numel()
                    total_escalated += escalate_mask.sum().item()
                    total_units += lf_data.size(0)

        average_loss = total_loss / total_units if total_units > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        usage = total_escalated / total_units if total_units > 0 else 0.0

        return {'loss': average_loss, 'accuracy': accuracy, 'usage': usage}