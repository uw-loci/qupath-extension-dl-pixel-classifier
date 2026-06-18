"""
Get model layer structure for freeze/unfreeze configuration.

Inputs:
    architecture: str
    encoder: str
    num_channels: int
    num_classes: int

Outputs:
    layers: list of dicts with name, display_name, param_count, is_encoder,
            depth, recommended_freeze, description
"""
import logging

logger = logging.getLogger("dlclassifier.appose.model_layers")

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: architecture, encoder, num_channels, num_classes

# --- Custom (non-SMP) architecture layer enumeration --------------------
# The package's PretrainedModelsService.get_model_layers() routes through an
# SMP arch_map that does NOT include tiny-unet or muvit, so it raised
# "Unknown architecture" and returned [] for them. The Java UI then fell back
# to generic resnet-style groups (encoder.layer_early/mid/late) that (a) show
# 0 params because tiny-unet has no encoder.* modules, and (b) freeze nothing
# (TinyUNet's params live under stem/enc/dec/head). We enumerate the model's
# real top-level blocks here so the freeze UI shows true param counts and the
# group names match named_parameters() prefixes -- the package's generic
# _freeze_layer() then freezes them correctly. Inlined in the JAR script (not
# delegated to the package) so it works even on an older installed package,
# same rationale as the inline normalization in inference_pixel.py.

_DEC_HINTS = ("dec", "up", "head", "decoder", "segmentation", "classifier",
              "final", "out", "logits")
_ENC_HINTS = ("stem", "enc", "encoder", "down", "backbone", "patch",
              "blocks", "stage", "embed")


def _classify_block(name):
    """Return True if the block looks encoder-side, False if decoder/head,
    or None when the name gives no hint (caller falls back to position)."""
    low = name.lower()
    if any(h in low for h in _DEC_HINTS):
        return False
    if any(h in low for h in _ENC_HINTS):
        return True
    return None


def _enumerate_top_level_blocks(model):
    """List a custom model's top-level child modules that own parameters as
    freeze groups whose 'name' matches named_parameters() prefixes."""
    children = [(n, m) for n, m in model.named_children()
                if sum(p.numel() for p in m.parameters()) > 0]
    total = len(children)
    layers = []
    for idx, (name, module) in enumerate(children):
        param_count = int(sum(p.numel() for p in module.parameters()))
        cls = _classify_block(name)
        is_encoder = cls if cls is not None else (idx < total - 1)
        type_name = type(module).__name__
        layers.append({
            "name": name,
            "display_name": ("Encoder: %s" if is_encoder else "Decoder: %s") % name,
            "param_count": param_count,
            "is_encoder": bool(is_encoder),
            "depth": idx,
            # Recommend freezing encoder-side blocks for transfer learning;
            # never the decoder/head (must adapt to the new task).
            "recommended_freeze": bool(is_encoder),
            "description": "%s block -- %s parameters" % (type_name, format(param_count, ",")),
        })
    return layers


def _build_custom_model(architecture, encoder, num_channels, num_classes):
    """Build a tiny-unet or muvit model for structure inspection, or None."""
    if architecture == "tiny-unet":
        from dlclassifier_server.models.tiny_unet import TinyUNet
        base, depth = 16, 4
        try:
            # backbone strings look like "compact-16x3" or "tiny-16x4" or "16x4"
            tail = str(encoder).split("-")[-1].lower()
            if "x" in tail:
                b, d = tail.split("x")
                base, depth = int(b), int(d)
        except (ValueError, AttributeError, IndexError):
            pass
        return TinyUNet(in_channels=num_channels, n_classes=num_classes,
                        base=base, depth=depth)
    if architecture == "muvit":
        from dlclassifier_server.services.muvit_model import create_muvit_model
        return create_muvit_model(
            architecture={"backbone": encoder},
            num_channels=num_channels,
            num_classes=num_classes)
    return None


try:
    layers = None
    if architecture in ("tiny-unet", "muvit"):
        try:
            model = _build_custom_model(
                architecture, encoder, num_channels, num_classes)
            if model is not None:
                layers = _enumerate_top_level_blocks(model)
        except Exception as ce:
            import traceback as _tb
            logger.warning(
                "Custom-architecture layer inspection failed for %s/%s: %s\n%s",
                architecture, encoder, ce, _tb.format_exc())
            layers = None

    if layers is None:
        from dlclassifier_server.services.pretrained_models import PretrainedModelsService
        pretrained = PretrainedModelsService()
        layers = pretrained.get_model_layers(
            architecture, encoder, num_channels, num_classes)

    task.outputs["layers"] = layers
    if not layers:
        # Empty list may indicate an upstream error that was caught and
        # logged inside get_model_layers(). Surface it so the Java side
        # can warn the user instead of silently using its local fallback.
        logger.warning(
            "get_model_layers returned 0 layers for architecture=%s encoder=%s "
            "num_channels=%d num_classes=%d. Java will fall back to built-in "
            "layer defaults. Check worker log above for the root cause.",
            architecture, encoder, num_channels, num_classes)
except Exception as e:
    # Log full traceback so a regression like the num_channels free-variable
    # bug is not silent when it recurs.
    import traceback as _tb
    logger.error("Failed to get model layers: %s\n%s", e, _tb.format_exc())
    task.outputs["layers"] = []
