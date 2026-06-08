# Context and Downsample: Design and Tile-Geometry Reference

Design/reference doc for how the **Resolution** ("downsample") control, the **Surrounding
context** ("context scale") control, and the derived **context padding** interact with training
tiles, the per-tile Training Area Issues preview, and production inference.

It exists because these three quantities have caused persistent "tile-effects" confusion, and
because the per-tile Training Area Issues display does not match what the model produces when
applied to the same image. This document is the canonical explanation of *why* the three code paths
differ and what the intended invariants are.

> **Status:** the geometry described in "Three code paths" below is the *current* (divergent)
> behavior. The "Target invariants" section is what we are converging on. When the unification fix
> lands, update the "Three code paths" tables so they describe one shared spec instead of three.

---

## 1. The three controls

| UI label (Train dialog) | Java field | Request key | Meaning |
|---|---|---|---|
| **Resolution:** | `TrainingConfig.downsample` (double) | `architecture["downsample"]` | Pixel size tiles are read at. 1x = full res; 4x = quarter res (more area per tile, less detail). |
| **Surrounding context:** | `TrainingConfig.contextScale` (int, default 1) | `architecture["context_scale"]` (sent only when > 1) | When > 1, a second "context" tile is read from a region `contextScale` times larger and downsampled to the tile's pixel dimensions, then concatenated on the channel axis (C -> 2C). |
| *(derived, not a UI field)* | `contextPadding` (export) / `inputPadding` + `contextInferencePad` (inference) | `config.json["context_padding"]` | A border of *real* image pixels around each tile so the model sees neighbours instead of zero-padding at tile edges. |

Data flow for the two UI controls:

```
TrainingDialog.downsampleCombo  --parseDownsample-->  TrainingConfig.downsample
TrainingDialog.contextScaleCombo --parseContextScale--> TrainingConfig.contextScale
        |                                                       |
        +----- ApposeClassifierBackend (architecture{}) --------+
                              |
                  Python architecture.get("downsample", 1.0)
                  Python architecture.get("context_scale", 1)
```

Key code:
- UI: `ui/TrainingDialog.java` (`downsampleCombo`, `contextScaleCombo`).
- Config: `model/TrainingConfig.java` (`getDownsample()`, `getContextScale()`).
- Serialization: `service/ApposeClassifierBackend.java` (`architecture.put("downsample", ...)`,
  `architecture.put("context_scale", ...)` when `> 1`).
- Padding (export): `controller/TrainingWorkflow.java` `computeTrainingContextPadding(tileSize, config)`
  which is exactly `InferenceConfig.computeEffectivePadding(tileSize, config.getOverlap())`.
- Padding (inference): `service/DLPixelClassifier.java` constructor (`inputPadding`,
  `contextInferencePad`).

---

## 2. The multi-scale context mechanism (channel layout)

When `contextScale > 1`, every tile becomes a **detail** tile plus a **context** tile, concatenated
along channels. A 3-channel RGB tile becomes a 6-channel model input: the first C channels are
detail, the next C are context.

**Channel layout is correct and consistent** between training and inference -- this was explicitly
verified, because it is an easy place to introduce a silent bug:

- **Training** (`python_server/.../services/training_service.py`, `SegmentationDataset.__getitem__`):
  `np.concatenate([img_array, ctx_array], axis=2)` on HWC arrays. Per pixel, the 2C values are laid
  out `[d0, d1, d2, c0, c1, c2]`; `transpose(2,0,1)` then gives channel order `[d0,d1,d2,c0,c1,c2]`.
- **Inference** (`utilities/TileEncoder.java` `interleaveContextChannels`): copies a whole detail
  channel block, then a whole context channel block, *per pixel* -> per-pixel `[d0,d1,d2,c0,c1,c2]`.
  numpy `frombuffer(...).reshape(H, W, 2C)` reproduces the identical layout, and the subsequent
  `transpose(2,0,1)` yields the same channel order as training.

The method name "interleave" is misleading -- it does **not** interleave per channel. It arranges
bytes so numpy's per-pixel reshape matches training's per-pixel concat. See the JavaDoc on
`interleaveContextChannels` for the rationale.

> A hypothesis that training used sequential channel concat while inference interleaved per channel
> was investigated and **rejected**. Channel ordering is not the cause of the per-tile-vs-production
> mismatch. Do not re-introduce this theory.

---

## 3. Three code paths over one model

The same trained weights are exercised by three different pipelines with different tile plumbing.

### 3a. Training
`python_server/.../services/training_service.py` `SegmentationDataset`.
- Consumes the **exported on-disk tiles**: `train/images`, `train/masks`, and (when `contextScale>1`)
  `train/context` (plus the `validation/` equivalents).
- Detail tile is whatever was exported (size `patchSize + 2*contextPadding`); context tile is the
  exported context tile, bilinear-resized to the detail dims if they differ.
- Normalizes the 2C array via the shared `normalize_image(img, input_config)`.

### 3b. Per-tile "Training Area Issues" preview
`src/main/resources/.../scripts/evaluate_tiles.py`, launched on demand from the "Review Training
Areas" button (`controller/TrainingWorkflow.java` -> `ApposeClassifierBackend.evaluateTiles`).
- Loads **`model.pt`** (PyTorch eager) -- `evaluate_tiles.py` ~lines 275-289.
- Runs it over the **same exported on-disk tiles** as training (same `SegmentationDataset`, same
  `context_dir`, same normalization, `augment=False`).
- Saves per-tile loss / disagreement / prediction PNGs at full (padded) tile size.
- `service/AnnotationAdjuster.computePreview` center-crops the padded maps to `patchSize` before
  creating objects (`cropToCenter`). A center-crop reminder banner is shown in
  `ui/TrainingAreaIssuesDialog.java`.

This path therefore reflects **training-time tile evaluation**, replayed through PyTorch on the
exported tiles -- not production inference.

### 3c. Production inference
`service/DLPixelClassifier.java` + `python_server/.../services/inference_service.py`.
- Loads **ONNX** (`model_static_bn.onnx` / `model_static.onnx` / `model.onnx`;
  `inference_service.py` ~1064-1066), falling back to `model.pt` only if ONNX is missing.
- Tiles are **freshly read from the live image**. The requested region is expanded by
  `inputPadding + contextInferencePad` per side (`expandRequest`), reflection-padded on image-edge
  tiles, and the context tile is read live by `DLPixelClassifier.readContextTile`.
- Output is `CENTER_CROP`-blended and cropped to the stride region (`cropArgmaxToStride`).

### Side-by-side

| Aspect | Training (3a) | Per-tile preview (3b) | Production inference (3c) |
|---|---|---|---|
| Runtime | PyTorch (`model.pt`) | PyTorch (`model.pt`) | **ONNX** |
| Tile source | exported on-disk tiles | exported on-disk tiles | **live image, re-read** |
| Detail tile px | `patchSize + 2*contextPadding` | same | `core + 2*(inputPadding + contextInferencePad)` |
| Context extent | `patchSize*contextScale` (full-res) | same | `tileSize*contextScale` (full-res) |
| Context resized to | `patchSize`, then detail dims | same | **expanded detail dims** |
| Channel order | `[detail.., context..]` | same | same (matches) |
| Crop | n/a | center-crop to `patchSize` for objects | center-crop to stride |

The preview (3b) is byte-identical to training (3a). It is **3c** that diverges from both.

---

## 4. Worked geometry example

> The first version of this section claimed the geometry always diverges. A precise read of
> the code shows it is **config-conditional**, and that the larger, overlap-independent
> divergence is the baked ONNX input shape. Corrected below.

Config: `tileSize = patchSize = 512`, `contextScale = 4`, `downsample = 1`. The relevant code:
- Export detail tile (`AnnotationExtractor.readPaddedTile`): `tileSize + 2*contextPadding`,
  `contextPadding = InferenceConfig.computeEffectivePadding(tileSize, overlap) = min(overlap, (tileSize-1)/2)`.
- Inference declares **stride** (`tileSize - 2*inputPadding`) as its tile size to QuPath, then
  `expandRequest` reads `stride + 2*(inputPadding + contextInferencePad)` =
  `tileSize + 2*contextInferencePad`. So the **inference model input = `tileSize + 2*contextInferencePad`**.
- `inputPadding = computeEffectivePadding(tileSize, overlap)` (same as `contextPadding`);
  `contextInferencePad = (tileSize/4)` rounded to a multiple of 32, added **only** when `contextScale > 1`.

So the model-input dimensions are:

| | Training / preview (3a, 3b) | Production inference (3c) |
|---|---|---|
| Model input (px) | `tileSize + 2*contextPadding` = `tileSize + 2*inputPadding` | `tileSize + 2*contextInferencePad` |
| Equal when... | -- | `contextInferencePad == inputPadding` |

- **overlap = 128** -> `inputPadding = 128`, `contextInferencePad = 128`. Both inputs = `768`.
  **Geometry matches.** (My earlier table wrote production as `core + 512`; with `core = stride = 256`
  that is also `768` -- it matches, it does not diverge.)
- **overlap = 64** -> `inputPadding = 64`, `contextInferencePad = 128`. Training input = `640`,
  inference input = `768`. Context FOV is `tileSize*contextScale = 2048` in both, stretched onto
  `640` vs `768` -> **the context relative scale diverges** (2048/640 = 3.2 vs 2048/768 = 2.67).

So the export-vs-inference geometry divergence appears **only when `inputPadding != contextInferencePad`**,
i.e. at smaller overlaps. For the recommended high-overlap config it does not occur, and the mismatch
the user observes must come from one of the other causes below.

### 4a. The overlap-independent divergence: baked ONNX input shape

`architecture["input_size"]` is set to the **unpadded** `[tileSize, tileSize]`
(`ApposeClassifierBackend.java:232`), and the static ONNX is exported with a dummy input of that
shape (`training_service.py:4943-4951`). But a `contextScale > 1` model is *trained* on the
**padded** `tileSize + 2*contextPadding` input. So `model_static.onnx` is baked at `512x512` while
the network actually learned on `768x768` tiles. At inference the expanded read is `768x768`, which
does **not** match the baked `512x512`, forcing `inference_service`'s static-shape fallback (drop to
the dynamic `model.onnx`, or center-crop/pad). This is independent of overlap and applies to every
context model.

---

## 5. Root-cause analysis of the per-tile-vs-production mismatch

Ranked by expected impact (to be confirmed with the Phase 0 harness, `tools/diagnose_eval_vs_inference.py`):

1. **Baked ONNX input shape (Section 4a).** Static ONNX baked at unpadded `tileSize` while the
   model trained on the padded size, so context-model inference takes the static-shape fallback /
   crop path instead of running the graph the model was traced/trained for. Overlap-independent.
2. **Runtime mismatch.** Preview uses PyTorch eager (`model.pt`); production uses ONNX. Even on a
   byte-identical input these can differ (ONNX export, static-shape handling, BatchNorm/BatchRenorm).
3. **Geometry mismatch (Section 4).** Detail/context relative scale diverges, but **only when
   `inputPadding != contextInferencePad`** (smaller overlaps). Not present in the recommended config.
4. **Normalization of the 2C array.** Confirm `normalize_image` applies the intended per-channel
   statistics to the doubled channel set identically on both sides (the same `input_config`).
5. **Displayed-vs-cropped maps.** Preview displays full padded maps (with a center-crop reminder)
   while production crops to stride. Minor.

**Rejected:** channel-ordering mismatch (Section 2).

A small diagnostic harness (dump the production model-input tensor vs the preview model-input tensor
for the same tile; diff `model.pt` vs ONNX logits on a byte-identical input) pins which of (1)-(3)
dominates. Record the measured result here when run.

---

## 6. Target invariants

The fix converges on two invariants:

- **Invariant A -- display == production.** The per-tile Training Area Issues prediction must be
  byte-faithful to what production inference yields for that tile's region. (Per the project decision,
  the preview is meant to be a faithful inference preview, not a separate training-tile diagnostic.)
- **Invariant B -- production == training.** Export-time and inference-time context + padding
  reconstruction must use **one shared spec**, so the model is fed at inference exactly the geometry
  it learned. Persist the spec into `metadata.json` so inference reads it rather than recomputing
  divergent values.

Once B holds and the preview uses the production runtime, A largely follows: replaying the exported
tiles through the production runtime equals reconstructing the same region live from the image.

When implemented, fold the three rows of the Section 3 table into one and update Section 4 to show a
single geometry.

---

## See also
- [Parameter Reference](PARAMETERS.md) -- Resolution / Context Scale fields and tooltips.
- [Inference Guide](INFERENCE_GUIDE.md) -- applying a classifier; context-scale display in the table.
- [Appose Dev Guide](APPOSE_DEV_GUIDE.md) -- constraints for the Python paths touched here.
