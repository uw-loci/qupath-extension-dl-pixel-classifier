# Documentation Image Registry

This file tracks the relationship between documentation screenshots and the Java UI classes that produce them. When a tracked Java file changes in a way that affects the rendered UI, the corresponding screenshot may need to be re-captured.

**How the check works:** `tools/check-doc-images.sh` parses the table below, then for each screenshot diffs its source Java file(s) from the screenshot's git commit (or its acknowledgement point, whichever is newer) to `HEAD`. It flags a screenshot only when the diff touches *UI-construction* code (new controls, layout additions, visible text such as `setText` / `setTitle` / `getChildren().add(...)`). Internal wiring, logging, and refactors are ignored, so refactors don't produce false positives. The `Last Verified` / `Status` columns are human notes; the check itself uses git history, not those columns.

**Maintenance:**
- Add a row when you add a screenshot; remove the row when you delete one.
- After re-capturing a screenshot, commit the new PNG -- that resets its baseline automatically.
- To accept a screenshot as current despite a source change you don't want to re-shoot: `tools/check-doc-images.sh --ack <Image.png> --note "why"` (records it in `IMAGE_ACKS.tsv`).
- Rows whose "source" is not a `.java` file (QuPath views, result composites, diagrams) are intentionally skipped by the check.

## Image-to-Source Mapping

| Screenshot | Java Source File(s) | Last Verified | Status |
|------------|--------------------|---------------|--------|
| `menu-dl-pixel-classifier.png` | `SetupDLClassifier.java` | 2026-07-22 | OK |
| `train-dialog-configure-classifier.png` | `ui/TrainingDialog.java` | 2026-07-22 | OK |
| `train-dialog-tiles-resolution-preview.png` | `ui/TrainingDialog.java` | 2026-07-22 | OK |
| `train-dialog-transfer-learning-layers.png` | `ui/TrainingDialog.java`, `ui/LayerFreezePanel.java` | 2026-07-22 | OK |
| `training-progress-dialog.png` | `ui/ProgressMonitorController.java` | 2026-07-22 | OK |
| `training-progress-charts.png` | `ui/ProgressMonitorController.java` | 2026-07-22 | OK |
| `training-loss-ohem-crossover.png` | `ui/ProgressMonitorController.java` | 2026-07-22 | OK |
| `training-area-issues-confusion-matrix.png` | `ui/TrainingAreaIssuesDialog.java` | 2026-07-22 | OK |
| `training-area-issues-loss-heatmap.png` | `ui/TrainingAreaIssuesDialog.java` | 2026-07-22 | OK |
| `manage-classifiers.png` | `controller/ModelManagementWorkflow.java` | 2026-07-22 | OK |
| `inference-tissue-prediction.png` | Inference result composite (before/after tissue predictions; not a single UI class) | 2026-07-22 | OK |
| `annotated-tissue-sparse-annotations.png` | QuPath annotation view (sparse line annotations; not a single UI class) | 2026-07-22 | OK |
| `project-image-list.png` | QuPath project image list (not a DL UI class) | 2026-07-22 | OK |

## Missing Screenshots (UI exists but no image)

Dialogs users interact with that would benefit from a figure but don't have one yet. Capture on the Windows workstation, add the PNG under `docs/images/`, and add a row above.

| UI Component | Java Source | Suggested Filename |
|-------------|------------|-------------------|
| Apply Classifier / Inference dialog | `ui/InferenceDialog.java` | `inference-dialog.png` |
| Advanced Augmentation dialog | `ui/AdvancedAugmentationDialog.java` | `train-dialog-advanced-augmentation.png` |
| Channel selection panel | `ui/ChannelSelectionPanel.java` | `train-dialog-channel-selection.png` |
| Setup DL Environment wizard | `ui/SetupEnvironmentDialog.java` | `setup-environment-wizard.png` |
| SSL / MAE pretraining dialogs | `ui/SSLPretrainingDialog.java`, `ui/MAEPretrainingDialog.java` | `pretraining-dialog.png` |
| Domain adaptation (AdaBN) dialog | `ui/AdaBNDialog.java` | `adabn-dialog.png` |
| Per-image Train/Val/Both roles + Auto Distribute | `ui/TrainingDialog.java` (Training Data Source section) | `train-dialog-split-roles.png` |
