import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SoftmaxResponseMethod:
    """LF-only "softmax response" selective-classification baseline: escalates
    to HF whenever the LF model's own softmax confidence falls below a
    threshold, chosen to hit a target coverage.

    Classification only - requires (B, num_classes) logits from both lf_model
    and hf_model; raises NotImplementedError if either model returns anything
    else (e.g. a segmentation map).

    Runs its own inference pass over lf_model/hf_model (val_dl/test_dl are the
    raw (lf_data, hf_data, label) cascade dataloaders) rather than reading a
    precomputed cache like the gate stage does - it needs the LF model's full
    per-class softmax distribution plus the true label to build its
    confidence-threshold curve, and the shared FE-stage cache
    (save_latent/FE_Dataset) no longer keeps either around (see
    helpers.save_latent's docstring: it only saves each fidelity's precomputed
    loss/correctness, not raw output/label, to stay affordable for
    spatial/segmentation outputs). This baseline is optional
    (run_comparisons=True only) and small (one pass per split), so paying its
    own inference cost here is cheap relative to that.
    """
    def __init__(self, lf_model, hf_model, val_dl, test_dl, device, train_body=True, hf_input_mode: str = "concat"):
        self.lf_model = lf_model.to(device)
        self.hf_model = hf_model.to(device)
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device
        self.train_body = train_body
        self.hf_input_mode = hf_input_mode
        self._test_cache = None  # (lf_output, hf_output, label), filled by get_softmax_thresholds

    def _lf_outputs_and_labels(self, dataloader):
        """val-only pass: threshold selection only needs LF's own confidence
        ranking, not HF (HF is only needed for test-time cascade accuracy)."""
        self.lf_model.eval()
        outputs, labels = [], []
        with torch.no_grad():
            for lf_data, _, label in dataloader:
                lf_data = lf_data.to(self.device, torch.float)
                head = self.lf_model(lf_data) if self.train_body else self.lf_model.head(lf_data)
                outputs.append(head["output"].cpu())
                labels.append(label)
        return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

    def _lf_hf_outputs_and_labels(self, dataloader):
        from helpers import assemble_hf_input

        self.lf_model.eval()
        self.hf_model.eval()
        lf_outputs, hf_outputs, labels = [], [], []
        with torch.no_grad():
            for lf_data, hf_data, label in dataloader:
                lf_data = lf_data.to(self.device, torch.float)
                hf_data = hf_data.to(self.device, torch.float)

                lf_head = self.lf_model(lf_data) if self.train_body else self.lf_model.head(lf_data)
                lf_out = lf_head["output"]
                if lf_out.dim() != 2:
                    raise NotImplementedError(
                        f"SoftmaxResponseMethod is classification-only; lf_model output has shape "
                        f"{tuple(lf_out.shape)}, expected (B, num_classes)"
                    )

                hf_input = assemble_hf_input(lf_data, hf_data, self.hf_input_mode, hf_model=self.hf_model)
                hf_head = self.hf_model(hf_input) if self.train_body else self.hf_model.head(hf_input)
                hf_out = hf_head["output"]
                if hf_out.dim() != 2:
                    raise NotImplementedError(
                        f"SoftmaxResponseMethod is classification-only; hf_model output has shape "
                        f"{tuple(hf_out.shape)}, expected (B, num_classes)"
                    )

                lf_outputs.append(lf_out.cpu())
                hf_outputs.append(hf_out.cpu())
                labels.append(label)
        return torch.cat(lf_outputs, dim=0), torch.cat(hf_outputs, dim=0), torch.cat(labels, dim=0)

    def get_softmax_thresholds(self, usage_list):
        """Returns (acc_vals, usage_vals), both dicts keyed by target usage
        (not by threshold - several usage targets can land on the exact same
        threshold once confidence saturates at 1.0, which used to silently
        overwrite entries in a threshold-keyed dict). acc_vals holds cascade
        accuracy (LF on admitted + HF on escalated), the same metric
        SelectiveNet/SAT report, for apples-to-apples Pareto comparison;
        selective-only accuracy and the chosen threshold are available via
        self.last_selective_acc / self.last_thresholds for callers that want
        them.
        """
        logger.info("\nRunning Softmax Response on LF+HF")

        # Cached so the per-usage-value threshold trials below don't each
        # re-run both models' forward pass over the whole test split.
        self._test_cache = self._lf_hf_outputs_and_labels(self.test_dl)

        val_output, val_label = self._lf_outputs_and_labels(self.val_dl)
        probs = F.softmax(val_output, dim=1)
        max_probs, val_preds = torch.max(probs, dim=1)
        val_correct = (val_preds == val_label)
        all_max_probs = max_probs.numpy()
        n_val = val_label.size(0)

        # sort descending for coverage mapping
        sorted_probs = np.sort(all_max_probs)[::-1]
        acc_vals = {}
        usage_vals = {}
        self.last_selective_acc = {}
        self.last_thresholds = {}
        # Realized val usage/selective-acc at the chosen threshold - same
        # "calibrate on val, report val+test" symmetry SAT/SelectiveNet's
        # calibrated variants use (SR has never run hf_model on val - its
        # threshold selection doesn't need HF at all - so there's no val
        # cascade accuracy to report here, only selective).
        self.last_val_usage = {}
        self.last_val_selective_acc = {}

        for target_usage in usage_list:
            coverage = 1 - target_usage

            if coverage <= 0:
                # usage=1: escalate everyone. No finite confidence threshold
                # admits nobody when ties sit exactly at the max probability
                # (sorted_probs[0]), so force it explicitly instead of
                # relying on the floor-index fallback to land on idx=0.
                threshold = float('inf')
            else:
                idx = max(0, int(np.floor(len(sorted_probs) * coverage)) - 1)
                threshold = float(sorted_probs[idx])

            selective_acc, cascade_acc, realized_usage = self.apply_softmax_threshold(threshold)

            val_admitted = max_probs >= threshold
            n_val_admitted = int(val_admitted.sum().item())
            val_usage = 1 - (n_val_admitted / n_val) if n_val > 0 else float('nan')
            val_selective_acc = (
                val_correct[val_admitted].float().mean().item() if n_val_admitted > 0 else float('nan')
            )

            logger.info(f"\n\tTarget usage {100*target_usage:.2f}% -> val threshold {threshold}")
            logger.info(
                f"\tRealized val usage: {100*val_usage:.2f}% | Realized test usage: {100*realized_usage:.2f}% | "
                f"Selective acc: {100*selective_acc:.2f}% | Cascade acc: {100*cascade_acc:.2f}%"
            )

            acc_vals[target_usage] = 100 * cascade_acc
            usage_vals[target_usage] = 100 * realized_usage
            self.last_selective_acc[target_usage] = 100 * selective_acc
            self.last_thresholds[target_usage] = threshold
            self.last_val_usage[target_usage] = 100 * val_usage
            self.last_val_selective_acc[target_usage] = 100 * val_selective_acc

        return acc_vals, usage_vals

    def apply_softmax_threshold(self, threshold):
        """Returns (selective_acc, cascade_acc, realized_usage) at the given
        confidence threshold, evaluated against the cached test outputs.
        selective_acc is LF-only accuracy on admitted (kept) samples;
        cascade_acc is the full-cascade accuracy (LF on admitted + HF on the
        rest) - the metric comparable to SelectiveNet/SAT. Ties at the
        threshold can admit more than the nominally-targeted coverage (e.g.
        many samples saturated at softmax confidence 1.0), so realized_usage
        is computed from the actual admit mask, not assumed to equal whatever
        target produced this threshold.
        """
        lf_output, hf_output, label = self._test_cache
        N = label.size(0)

        lf_probs = F.softmax(lf_output, dim=1)
        lf_max_probs, lf_preds = torch.max(lf_probs, dim=1)
        lf_correct = (lf_preds == label)

        hf_preds = torch.argmax(hf_output, dim=1)
        hf_correct = (hf_preds == label)

        admitted_mask = lf_max_probs >= threshold
        n_admitted = int(admitted_mask.sum().item())

        selective_acc = (
            lf_correct[admitted_mask].float().mean().item() if n_admitted > 0 else float('nan')
        )
        cascade_correct = lf_correct[admitted_mask].sum() + hf_correct[~admitted_mask].sum()
        cascade_acc = (cascade_correct.item() / N) if N > 0 else float('nan')
        realized_usage = 1 - (n_admitted / N) if N > 0 else float('nan')

        return selective_acc, cascade_acc, realized_usage

class SelectiveNetMethod:
    def __init__(self, model_folder, alpha=0.5, c=0.8, lmda=32):
        self.model_folder = model_folder
        self.alpha = alpha
        self.c = c
        self.lmda = lmda

    def selective_loss_class(self, y_true, selection_logits, lf_logits, lamda=32, c=0.8):
        # Paper's selective risk is mean(g*ce)/mean(g), not mean(g*ce) alone -
        # without dividing by coverage, driving coverage down (which the
        # penalty term is constantly pushing for whenever coverage > c)
        # shrinks the risk term for free, independent of whether the
        # remaining selected samples are actually easy/correct.
        selection_prob = torch.sigmoid(selection_logits).view(-1)
        ce = F.cross_entropy(lf_logits, y_true, reduction='none')

        coverage = selection_prob.mean()
        selective_risk = (selection_prob * ce).mean() / (coverage + 1e-8)

        penalty = lamda * torch.clamp(-coverage + c, min=0) ** 2
        return selective_risk + penalty

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
        if hf_model is not None:
            hf_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0  # For classification: samples; for segmentation: pixels
        total_selected_images = 0  # Images selected by SelectiveNet
        total_coverage = 0.0
        total_images = 0  # real running count of images/samples seen, regardless of branch -
        # replaces len(dataloader)*dataloader.batch_size below, which overcounts
        # whenever the last batch is a partial one.
        total_selective_correct = 0  # LF-only correctness on kept samples (selective, not cascade, accuracy)
        total_coverage_weighted = 0.0  # classification only - sample-weighted (not batch-averaged) coverage

        is_segmentation = None

        context_manager = torch.no_grad() if not is_training else torch.enable_grad()

        with context_manager:
            for lf_data, hf_data, target in dataloader:
                total_images += lf_data.size(0)
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
                    total_coverage_weighted += coverage * lf_data.size(0)
                    
                    # SelectiveNet predictions
                    sel_pred = torch.argmax(selnet_output, dim=1)  # [B]
                    sel_correct = (sel_pred[cascade_mask] == target[cascade_mask]).sum().item()

                    # HF model on abstained samples - no_grad regardless of is_training,
                    # since HF is frozen here and was otherwise building an unused
                    # autograd graph (and holding onto its activations) during every
                    # training-mode batch.
                    hf_correct = 0
                    abstained_mask = ~cascade_mask
                    if hf_model is not None and abstained_mask.sum() > 0:
                        with torch.no_grad():
                            abstained_data = assemble_hf_input(lf_data[abstained_mask], hf_data[abstained_mask], hf_input_mode)
                            hf_layers = hf_model(abstained_data) if train_body else hf_model.head(abstained_data)
                            hf_pred = torch.argmax(hf_layers["output"], dim=1)
                            hf_correct = (hf_pred == target[abstained_mask]).sum().item()

                    total_correct += sel_correct + hf_correct
                    total_selective_correct += sel_correct
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
        
        n_batches = len(dataloader)
        avg_loss = total_loss / max(total_images, 1)
        # Classification: sample-weighted (fixed - was an unweighted per-batch
        # average, which biases the result whenever batch sizes vary, e.g. a
        # partial last batch). Segmentation: left as the pre-existing
        # per-batch average - out of scope for this classification-only pass
        # (see this file's SelectiveNet/SAT fixes: only the classification
        # branches were touched, segmentation left as-is throughout).
        avg_coverage = (
            total_coverage_weighted / max(total_images, 1) if not is_segmentation
            else total_coverage / max(n_batches, 1)
        )
        accuracy = total_correct / total_samples if total_samples > 0 else 0  # cascade accuracy

        # For segmentation: this is % of images routed to SelectiveNet
        # For classification: this is % of samples routed to SelectiveNet
        selection_rate = total_selected_images / max(total_images, 1)
        usage = 1 - selection_rate

        # LF-only accuracy on the samples SelectiveNet actually kept (not the
        # cascade accuracy above) - classification only; segmentation doesn't
        # track a comparable per-sample selective count, so this stays NaN for
        # that branch rather than guessing at a definition for it. NaN (not 0)
        # at total_selected_images==0 too - "0% accurate" would misreport an
        # undefined quantity (nothing was kept to be accurate or inaccurate
        # about) as if the model were maximally wrong.
        selective_accuracy = (
            total_selective_correct / total_selected_images
            if (not is_segmentation and total_selected_images > 0) else float('nan')
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,  # cascade accuracy: LF on kept + HF on escalated
            'selective_accuracy': selective_accuracy,  # LF-only accuracy on kept samples
            'coverage': avg_coverage,  # Average pixel-level selection probability
            'selection_rate': selection_rate,  # % of images using SelectiveNet
            'usage': usage
        }

    def get_selection_probs(self, model, dataloader, train_body=True):
        """Classification only. Forward-only pass returning each sample's
        selection probability g = sigmoid(select) as a 1D numpy array, in the
        dataloader's own iteration order - same contract as
        SelfAdaptiveTrainingMethod.get_abstain_probs (callers must pass a
        non-shuffled loader for that order to mean anything downstream)."""
        model.eval()
        device = next(model.parameters()).device
        probs_out = []
        with torch.no_grad():
            for lf_data, _, _ in dataloader:
                lf_data = lf_data.to(device, torch.float)
                layers = model.selective_forward(lf_data) if train_body else model.selective_head(lf_data)
                select = layers["select"]
                if select.dim() != 2 or select.shape[1] != 1:
                    raise NotImplementedError(
                        f"get_selection_probs is classification-only; select has shape {tuple(select.shape)}, "
                        f"expected (B, 1)"
                    )
                probs_out.append(torch.sigmoid(select).squeeze(1).cpu())
        return torch.cat(probs_out, dim=0).numpy()

    @staticmethod
    def calibrate_threshold(val_g, target_usage):
        """threshold such that roughly target_usage of val's selection-
        probability distribution g falls below it (one_run escalates when
        `g < threshold`, i.e. cascade_mask = g >= threshold keeps a sample) -
        the SelectiveNet-equivalent of SelfAdaptiveTrainingMethod.calibrate_tau,
        same edge-case rationale: target_usage<=0 forces threshold=-inf
        (g > -inf always, so cascade_mask is always True - nobody escalates,
        regardless of ties at val's min) and target_usage>=1 forces
        threshold=+inf (g >= inf is never true - everybody escalates,
        regardless of ties at val's max)."""
        if target_usage >= 1:
            return float('inf')  # escalate everyone
        if target_usage <= 0:
            return float('-inf')  # escalate no one
        return float(np.quantile(val_g, target_usage))

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
        # Guards against training silently proceeding with every target still
        # at its zero-init value (sat_loss would then see t_scalar=0 for every
        # sample, i.e. loss = -log(p_abstain) only - a real loss that trains
        # *something*, but not the intended one, so a forgotten
        # initialize_targets() call wouldn't raise, just quietly degrade).
        self._initialized = False

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
        self._initialized = True

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
        if is_training:
            assert self._initialized, (
                "SelfAdaptiveTrainingMethod.initialize_targets() must be called before the first "
                "training one_run() - otherwise every self.targets row is still its zero-init "
                "value and sat_loss silently degrades to -log(p_abstain) for every sample."
            )
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
        total_selective_correct = 0.0  # LF-only accuracy on kept (non-escalated) samples
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
                    kept_correct = (own_pred[kept_mask] == target[kept_mask]).sum().item()
                    correct = kept_correct

                    # HF is frozen here regardless of is_training - no_grad avoids
                    # building an unused autograd graph through it on every
                    # training-mode batch.
                    if hf_model is not None and escalate_mask.sum() > 0:
                        with torch.no_grad():
                            escalated_data = assemble_hf_input(lf_data[escalate_mask], hf_data[escalate_mask], hf_input_mode)
                            hf_layers = hf_model(escalated_data) if train_body else hf_model.head(escalated_data)
                            hf_pred = torch.argmax(hf_layers["output"], dim=1)
                            correct += (hf_pred == target[escalate_mask]).sum().item()

                    total_correct += correct
                    total_selective_correct += kept_correct
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
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0  # cascade accuracy
        kept = total_units - total_escalated
        # Classification-only (see total_selective_correct's comment above);
        # segmentation, and the kept==0 case (nothing to be accurate or
        # inaccurate about - distinct from "0% accurate"), both report NaN
        # rather than a 0 that would misreport an undefined quantity.
        selective_accuracy = (
            total_selective_correct / kept if (not self.is_segmentation and kept > 0) else float('nan')
        )
        usage = total_escalated / total_units if total_units > 0 else 0.0

        return {
            'loss': average_loss,
            'accuracy': accuracy,  # cascade accuracy: own-head on kept + HF on escalated
            'selective_accuracy': selective_accuracy,  # own-head accuracy on kept samples only
            'usage': usage,
        }

    def get_abstain_probs(self, model, dataloader, train_body=True):
        """Classification only. Runs a forward-only pass (no HF, no loss, no
        target updates) over dataloader and returns each sample's abstain
        probability as a 1D numpy array, in the dataloader's own iteration
        order - callers must pass a non-shuffled loader (e.g. the plain
        val/test IndexedDataset loaders main.py already builds for one_run)
        for that order to mean anything downstream.
        """
        if self.is_segmentation:
            raise NotImplementedError("get_abstain_probs is classification-only")

        model.eval()
        device = next(model.parameters()).device
        probs_out = []
        with torch.no_grad():
            for _, lf_data, _, _ in dataloader:
                lf_data = lf_data.to(device, torch.float)
                logits = model(lf_data)["output"] if train_body else model.head(lf_data)["output"]
                probs = F.softmax(logits, dim=1)
                probs_out.append(probs[:, self.num_classes].cpu())
        return torch.cat(probs_out, dim=0).numpy()

    @staticmethod
    def calibrate_tau(val_abstain_probs, target_usage):
        """tau such that roughly target_usage of val's abstain-probability
        distribution exceeds it (escalation is `abstain_prob > tau`) - the
        SAT-equivalent of SelectiveNet/SR's val-calibrated threshold, used in
        place of a single fixed tau=0.5 so SAT is actually evaluated at each
        requested usage_list target rather than whatever usage 0.5 happens to
        produce.

        target_usage<=0 and >=1 are handled explicitly rather than left to
        the quantile: escalation uses strict `>`, so tau=quantile(...,1)
        (=max(val_abstain_probs)) wouldn't reliably escalate nobody (a test
        sample could exceed val's max) and tau=quantile(...,0)
        (=min(val_abstain_probs)) wouldn't reliably escalate everybody (ties
        at the min fail `>`) - +-inf make both guarantees exact regardless of
        the val/test distributions, mirroring SoftmaxResponseMethod's same
        usage<=0/>=1 edge handling.
        """
        if target_usage >= 1:
            return float('-inf')  # escalate everyone
        if target_usage <= 0:
            return float('inf')  # escalate no one
        return float(np.quantile(val_abstain_probs, 1 - target_usage))