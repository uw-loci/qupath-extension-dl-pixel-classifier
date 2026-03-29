"""Training diagnostics: automated checks on training health.

Run during training (periodically), at pause, or at completion to detect
common problems and warn the user via the training log.

Each check receives the training history and returns a list of warnings.
Warnings are logged once and not repeated on subsequent checks.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TrainingDiagnostics:
    """Collects and runs diagnostic checks on training history.

    Usage::

        diag = TrainingDiagnostics(classes=["Gland", "Ignore*", "Normal"])
        # During training, periodically:
        diag.run_checks(training_history)
        # At pause or completion:
        diag.run_all_checks(training_history)
    """

    def __init__(self, classes: List[str]):
        self.classes = classes
        self._warned: set = set()  # track which warnings have been issued

    def run_checks(self, history: List[Dict[str, Any]],
                   min_epochs: int = 15) -> List[str]:
        """Run periodic checks (lightweight, called every N epochs).

        Args:
            history: training_history list from _run_training
            min_epochs: minimum epochs before running checks

        Returns:
            List of warning messages (empty if no issues)
        """
        if len(history) < min_epochs:
            return []

        warnings = []
        warnings.extend(self._check_val_outlier(history))
        warnings.extend(self._check_majority_collapse(history))
        warnings.extend(self._check_training_stall(history))
        return warnings

    def run_all_checks(self, history: List[Dict[str, Any]]) -> List[str]:
        """Run all checks (called at pause or completion).

        Args:
            history: training_history list from _run_training

        Returns:
            List of warning messages (empty if no issues)
        """
        if len(history) < 5:
            return []

        warnings = []
        warnings.extend(self._check_val_outlier(history))
        warnings.extend(self._check_majority_collapse(history))
        warnings.extend(self._check_training_stall(history))
        warnings.extend(self._check_class_never_learned(history))
        warnings.extend(self._check_loss_divergence(history))
        warnings.extend(self._check_lr_too_high(history))
        warnings.extend(self._check_loss_spikes(history))
        warnings.extend(self._check_class_weight_imbalance(history))
        warnings.extend(self._check_val_better_than_train(history))
        return warnings

    def _warn_once(self, key: str, message: str) -> Optional[str]:
        """Issue a warning only once per key."""
        if key in self._warned:
            return None
        self._warned.add(key)
        logger.warning("TRAINING DIAGNOSTIC: %s", message)
        return message

    # ==================== Individual Checks ====================

    def _check_val_outlier(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect validation outlier pattern: a class has loss spikes that
        cluster around a consistent value in >30% of epochs.

        This suggests one annotation region landed entirely in the validation
        set and the model never trains on that visual pattern.
        """
        warnings = []
        for cls_name in self.classes:
            losses = []
            for entry in history:
                pcl = entry.get("per_class_loss", {})
                if cls_name in pcl:
                    losses.append(pcl[cls_name])

            if len(losses) < 10:
                continue

            # Compute median and identify spikes (> 3x median)
            sorted_losses = sorted(losses)
            median = sorted_losses[len(sorted_losses) // 2]
            if median <= 0:
                continue

            spikes = [v for v in losses if v > median * 3]
            if len(spikes) < 3:
                continue

            spike_ratio = len(spikes) / len(losses)
            if spike_ratio < 0.25:
                continue

            # Check if spikes cluster around a consistent value
            spike_mean = sum(spikes) / len(spikes)
            spike_variance = sum((s - spike_mean) ** 2 for s in spikes) / len(spikes)
            spike_std = spike_variance ** 0.5
            coefficient_of_variation = spike_std / spike_mean if spike_mean > 0 else 1.0

            if coefficient_of_variation < 0.3:  # spikes are consistent
                msg = (
                    f"Class '{cls_name}' has consistent validation loss spikes "
                    f"(~{spike_mean:.1f}) in {spike_ratio*100:.0f}% of epochs. "
                    f"This often means one annotation region is entirely in the "
                    f"validation set. Fix: break large annotations of this class "
                    f"into smaller pieces, or add annotations on more images. "
                    f"See Troubleshooting guide for details."
                )
                w = self._warn_once(f"val_outlier_{cls_name}", msg)
                if w:
                    warnings.append(w)

        return warnings

    def _check_majority_collapse(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect majority-class collapse: multiple minority classes
        simultaneously drop to near-zero IoU while the majority class
        stays high.
        """
        warnings = []
        if len(history) < 10:
            return warnings

        # Find the majority class (highest median IoU)
        class_median_ious = {}
        for cls_name in self.classes:
            ious = [e.get("per_class_iou", {}).get(cls_name, 0)
                    for e in history]
            sorted_ious = sorted(ious)
            class_median_ious[cls_name] = sorted_ious[len(sorted_ious) // 2]

        majority_class = max(class_median_ious, key=class_median_ious.get)
        minority_classes = [c for c in self.classes if c != majority_class]

        if not minority_classes:
            return warnings

        # Count epochs where all minority classes collapse
        collapse_count = 0
        recent = history[-min(len(history), 20):]
        for entry in recent:
            pci = entry.get("per_class_iou", {})
            majority_ok = pci.get(majority_class, 0) > 0.5
            minorities_collapsed = all(
                pci.get(c, 0) < 0.05 for c in minority_classes
            )
            if majority_ok and minorities_collapsed:
                collapse_count += 1

        if collapse_count >= 3:
            msg = (
                f"Majority-class collapse detected: {collapse_count} of the "
                f"last {len(recent)} epochs had all minority classes "
                f"({', '.join(minority_classes)}) near zero IoU while "
                f"'{majority_class}' stayed high. This indicates the model "
                f"periodically predicts everything as '{majority_class}'. "
                f"Consider: lower learning rate, higher class weights for "
                f"minority classes, or switch to ReduceOnPlateau scheduler."
            )
            w = self._warn_once("majority_collapse", msg)
            if w:
                warnings.append(w)

        return warnings

    def _check_training_stall(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect training stall: train_loss hasn't improved in 20+ epochs."""
        warnings = []
        if len(history) < 25:
            return warnings

        recent_losses = [e.get("train_loss", 0) for e in history[-20:]]
        older_losses = [e.get("train_loss", 0) for e in history[-25:-20]]

        if not recent_losses or not older_losses:
            return warnings

        recent_avg = sum(recent_losses) / len(recent_losses)
        older_avg = sum(older_losses) / len(older_losses)

        # If recent average is >= older average, training has stalled
        if recent_avg >= older_avg * 0.99 and older_avg > 0:
            msg = (
                f"Training loss has not improved in the last 20 epochs "
                f"(avg {recent_avg:.4f} vs {older_avg:.4f} earlier). "
                f"The model may have converged or the learning rate may "
                f"need to be reduced. With ReduceOnPlateau, the scheduler "
                f"should reduce LR automatically; with OneCycleLR, the "
                f"cosine decay phase may not have started yet."
            )
            w = self._warn_once("training_stall", msg)
            if w:
                warnings.append(w)

        return warnings

    def _check_class_never_learned(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect a class that never achieves meaningful IoU (always < 0.1)."""
        warnings = []
        if len(history) < 20:
            return warnings

        for cls_name in self.classes:
            ious = [e.get("per_class_iou", {}).get(cls_name, 0)
                    for e in history]
            max_iou = max(ious) if ious else 0

            if max_iou < 0.1:
                msg = (
                    f"Class '{cls_name}' has never exceeded 0.10 IoU across "
                    f"{len(history)} epochs (best: {max_iou:.3f}). This class "
                    f"may have insufficient annotations, or its visual "
                    f"features may be too similar to another class. Check: "
                    f"(1) annotation quality and quantity, (2) class weights, "
                    f"(3) whether this class should be merged with another."
                )
                w = self._warn_once(f"never_learned_{cls_name}", msg)
                if w:
                    warnings.append(w)

        return warnings

    def _check_loss_divergence(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect train/val loss divergence (overfitting signal)."""
        warnings = []
        if len(history) < 20:
            return warnings

        recent = history[-10:]
        train_losses = [e.get("train_loss", 0) for e in recent]
        val_losses = [e.get("val_loss", 0) for e in recent]

        if not train_losses or not val_losses:
            return warnings

        avg_train = sum(train_losses) / len(train_losses)
        avg_val = sum(val_losses) / len(val_losses)

        # If val loss is > 2x train loss consistently, likely overfitting
        if avg_train > 0 and avg_val > avg_train * 2.0:
            msg = (
                f"Possible overfitting: validation loss ({avg_val:.4f}) is "
                f"{avg_val/avg_train:.1f}x the training loss ({avg_train:.4f}) "
                f"over the last 10 epochs. Consider: more augmentation, "
                f"more training data, freeze more encoder layers, or reduce "
                f"model capacity."
            )
            w = self._warn_once("loss_divergence", msg)
            if w:
                warnings.append(w)

        return warnings

    def _check_lr_too_high(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect learning rate too high: train loss oscillates wildly
        instead of decreasing smoothly, or increases over time.

        Common with OneCycleLR during ramp-up, or when the LR finder
        suggests a value that's too aggressive.
        """
        warnings = []
        if len(history) < 10:
            return warnings

        recent = [e.get("train_loss", 0) for e in history[-10:]]
        if not recent or recent[0] == 0:
            return warnings

        # Check if loss is increasing over the last 10 epochs
        first_half = sum(recent[:5]) / 5
        second_half = sum(recent[5:]) / 5
        if second_half > first_half * 1.2 and first_half > 0:
            msg = (
                f"Training loss is increasing ({first_half:.4f} -> "
                f"{second_half:.4f} over last 10 epochs). The learning "
                f"rate may be too high. With OneCycleLR, this can happen "
                f"during the ramp-up phase and may resolve during the "
                f"decay phase. With ReduceOnPlateau, try a lower initial "
                f"learning rate (e.g., 0.00005 instead of 0.0001)."
            )
            w = self._warn_once("lr_too_high", msg)
            if w:
                warnings.append(w)

        # Check for high variance (oscillating loss)
        mean_loss = sum(recent) / len(recent)
        variance = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
        cv = (variance ** 0.5) / mean_loss if mean_loss > 0 else 0

        if cv > 0.3:  # coefficient of variation > 30%
            msg = (
                f"Training loss is oscillating heavily (CV={cv:.0%} over "
                f"last 10 epochs). This suggests the learning rate is "
                f"too high for stable convergence. If using OneCycleLR, "
                f"this may improve during the decay phase. Otherwise, "
                f"reduce the learning rate."
            )
            w = self._warn_once("lr_oscillation", msg)
            if w:
                warnings.append(w)

        return warnings

    def _check_loss_spikes(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect sudden loss spikes mid-training (common in segmentation).

        A sudden spike followed by recovery is usually benign (one bad batch).
        Repeated spikes suggest instability.
        """
        warnings = []
        if len(history) < 15:
            return warnings

        val_losses = [e.get("val_loss", 0) for e in history]
        sorted_vals = sorted(val_losses)
        median_val = sorted_vals[len(sorted_vals) // 2]

        if median_val <= 0:
            return warnings

        # Count extreme spikes (> 5x median)
        spike_count = sum(1 for v in val_losses if v > median_val * 5)
        spike_ratio = spike_count / len(val_losses)

        if spike_count >= 3 and spike_ratio > 0.1:
            msg = (
                f"Validation loss has {spike_count} extreme spikes "
                f"(>{median_val*5:.2f}, which is 5x the median) across "
                f"{len(val_losses)} epochs ({spike_ratio:.0%}). Occasional "
                f"spikes are normal, but frequent spikes suggest instability. "
                f"Consider: lower learning rate, larger batch size, or "
                f"gradient accumulation."
            )
            w = self._warn_once("loss_spikes", msg)
            if w:
                warnings.append(w)

        return warnings

    def _check_class_weight_imbalance(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect when one class dominates with near-perfect IoU while
        others struggle, suggesting class weights need adjustment.
        """
        warnings = []
        if len(history) < 15:
            return warnings

        # Compute median IoU per class over recent epochs
        recent = history[-min(len(history), 20):]
        class_median_ious = {}
        for cls_name in self.classes:
            ious = sorted([e.get("per_class_iou", {}).get(cls_name, 0)
                           for e in recent])
            class_median_ious[cls_name] = ious[len(ious) // 2]

        best_class = max(class_median_ious, key=class_median_ious.get)
        worst_class = min(class_median_ious, key=class_median_ious.get)

        best_iou = class_median_ious[best_class]
        worst_iou = class_median_ious[worst_class]

        # Large gap between best and worst class
        if best_iou > 0.8 and worst_iou < 0.3 and best_iou - worst_iou > 0.5:
            msg = (
                f"Large IoU gap between classes: '{best_class}' median "
                f"IoU={best_iou:.2f} vs '{worst_class}' median "
                f"IoU={worst_iou:.2f}. The model is learning "
                f"'{best_class}' well but struggling with "
                f"'{worst_class}'. Consider: increase the class weight "
                f"for '{worst_class}' using the Rebalance Classes button, "
                f"add more annotations for '{worst_class}', or use "
                f"Focus Class to optimize for '{worst_class}' specifically."
            )
            w = self._warn_once("class_imbalance", msg)
            if w:
                warnings.append(w)

        return warnings

    def _check_val_better_than_train(self, history: List[Dict[str, Any]]) -> List[str]:
        """Detect validation loss consistently lower than training loss.

        This can indicate: data leakage, augmentation only applied to
        training (expected and normal), or dropout/regularization effects.
        Only warn if the gap is very large.
        """
        warnings = []
        if len(history) < 15:
            return warnings

        recent = history[-10:]
        count_val_better = sum(
            1 for e in recent
            if e.get("val_loss", 999) < e.get("train_loss", 0) * 0.5
        )

        if count_val_better >= 8:
            avg_train = sum(e.get("train_loss", 0) for e in recent) / len(recent)
            avg_val = sum(e.get("val_loss", 0) for e in recent) / len(recent)
            msg = (
                f"Validation loss ({avg_val:.4f}) is consistently much "
                f"lower than training loss ({avg_train:.4f}) over the last "
                f"10 epochs. A small gap is normal (augmentation is only "
                f"applied during training). A large gap may indicate: "
                f"(1) the validation set is too easy (not representative), "
                f"(2) data leakage between train/val splits, or "
                f"(3) very aggressive augmentation making training harder."
            )
            w = self._warn_once("val_better_than_train", msg)
            if w:
                warnings.append(w)

        return warnings
