# Entry and exit points for the DL Pixel Classifier.
#
# Rendered by tools/render_workflow_map.py -- EDIT THIS FILE, NOT THE PNG.
#   python3 tools/render_workflow_map.py            # re-render the figure
#   python3 tools/render_workflow_map.py --check     # verify labels against the source
#
# Every `label` and `menu` value below is real UI text. `--check` cross-references
# them against strings.properties, SetupDLClassifier.java, ClassifierHandler.java
# and InferenceConfig.java, so a renamed menu item shows up as a failed check
# instead of a quietly wrong figure.

[meta]
title = Where you can start, and where you come out
subtitle = Five ways into training, one route that skips it, three ways out.
caption = Diagram, not a screenshot. Menu paths and labels are exact; layout is drawn.
footer = Extensions > DL Pixel Classifier

[prereq]
label = Setup DL Environment...
note = once per install, before anything else

# ---------------------------------------------------------------- what you have

[have.unlabeled]
label = Unlabeled images
note = a project, nothing annotated

[have.annotated]
label = A few annotations
note = sparse brush strokes per class, saved to the project -- not just on screen

[have.model]
label = An existing classifier
note = trained here earlier, or shared

# ------------------------------------------------- optional: pretrain an encoder

[prep.mae]
label = MAE Pretrain Encoder...
menu = Utilities
note = self-supervised, no labels needed

[prep.ssl]
label = SSL Pretrain Encoder...
menu = Utilities
note = hidden by default (developer preference)
flag = dlclassifier.showDeveloperPretrainingOptions

# ------------------------------------------------------------------------- train

[train]
label = Train DL Pixel Classifier...
note = Weight Initialization -- pick one

[init.pretrained]
label = Use pretrained backbone weights
tag = ImageNet, histology, foundation
default = yes

[init.mae]
label = Use MAE pretrained encoder
tag = from MAE Pretrain Encoder...

[init.ssl]
label = Use SSL pretrained encoder
tag = from SSL Pretrain Encoder...

[init.continue]
label = Continue training from saved model
tag = from a classifier you already have
note = locks architecture, backbone, tile size, downsample

[init.scratch]
label = Train from scratch
tag = random init -- tiny-unet, fluorescence

# ------------------------------------------------------------------- the model

[model]
label = A trained classifier
note = Manage Classifiers... copies it to a production project

# ------------------------------------------------------------------------- apply

[apply]
label = Apply DL Pixel Classifier...
note = choose an output type

[out.measurements]
label = Measurements
note = added to parent annotations

[out.objects]
label = Objects
note = detections or annotations

[out.overlay]
label = Overlay
note = for looking; Toggle Prediction Overlay shows it live

# ----------------------------------------------------------------- the bypass

[adabn]
chip = no retraining
label = Calibrate model to current image...
menu = Utilities
note = recomputes BatchNorm statistics on the new images (AdaBN), using the model you already trained

# ------------------------------------------------------------------------- edges

[edge.unlabeled_to_mae]
text = unlabeled tiles

[edge.prep_to_init]
text = encoder weights

[edge.annotated_to_train]
text = from the project's SAVED data

[edge.model_to_continue]
text = all weights

[edge.train_to_model]
text = trains

[edge.model_to_apply]
text = pick a model

[edge.loop]
text = annotate more, train again

[edge.adabn_in]
text = new scanner, stain or batch

[edge.adabn_out]
text = seconds, no retraining

# ------------------------------------------------------------------------ legend

[legend.plain]
text = supervised path

[legend.accent]
text = skips training entirely

[legend.dashed]
text = hidden behind a preference
