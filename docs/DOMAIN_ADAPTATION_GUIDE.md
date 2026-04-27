# When Your Classifier Doesn't Work on New Images

## The Problem

You trained a pixel classifier on your fly wing images and it works great. Then a colleague sends you their fly wing images -- taken on a different microscope, with different lighting, or saved with different compression -- and your classifier performs poorly. The veins look washed out, the background is a different shade, or the contrast is just different enough that the model gets confused.

You don't have time to annotate hundreds of their images from scratch. And you shouldn't have to -- you already taught the model what veins, hairs, and intervein regions look like. The problem isn't that the model doesn't know what a vein is. The problem is that the new images *look* different enough that the model doesn't recognize them.

## What SSL Pretraining Does (Simple Explanation)

Your trained classifier has two parts:

1. **The encoder** (the "eyes") -- looks at pixels and extracts visual features like edges, textures, and patterns
2. **The decoder** (the "brain") -- takes those features and decides "this pixel is a vein, this pixel is background"

When your classifier fails on new images, it's usually because the encoder learned features specific to *your* microscope's images. It knows what a vein looks like in *your* images, but the same vein looks different enough in your colleague's images that the encoder doesn't produce the right features.

**SSL (Self-Supervised Learning) pretraining** lets the encoder practice looking at the new images without needing any annotations. It learns "what do images from this microscope look like?" by solving a puzzle: the model is shown two randomly altered versions of the same image and has to figure out they came from the same source. Through this process, the encoder adapts to the new image style.

## Step-by-Step: Adapting Your Classifier

### What You Need

- Your trained classifier (the `.pt` model file)
- A QuPath project with the new (unannotated) images loaded
- A small number of annotations on the new images (you'll add these at the end -- even just 2-3 images is often enough)

### Step 1: Mark the tissue regions

Open the new images in QuPath and draw rough annotations around the tissue areas. These don't need to be precise -- they just tell the system "extract tiles from here, not from the empty slide background." Any annotation class works (Tissue, Region, or whatever you use).

You do NOT need to label veins, hairs, etc. at this stage. Just outline where the tissue is.

### Step 2: Run SSL Pretrain Encoder

1. Go to **DL Pixel Classifier > Utilities > SSL Pretrain Encoder...**
2. **SSL Method**: Choose BYOL (works better with smaller datasets)
3. **Backbone**: Select the same backbone your original classifier used (check your model's `metadata.json` -- look at `architecture.backbone`, e.g., `resnet34`)
4. **Initialize from trained model**: Click Browse and select your existing trained `model.pt` file. This is the key step -- instead of starting from scratch, the encoder starts with everything it already learned from your images
5. **Data section**: Select the new images and check the annotation classes you used to mark tissue regions
6. **Epochs**: 100 is usually enough
7. Click **Start Pretraining**

This will take a few minutes to an hour depending on how many images you have and your GPU. The encoder is learning to "see" the new images while keeping its knowledge of what biological features look like.

### Step 3: Train with a few annotations

Now you need a small amount of ground truth on the new images. Annotate veins, hairs, and background on **2-5 images** from the new set. This is much less work than annotating everything from scratch.

1. Go to **DL Pixel Classifier > Train**
2. Choose the same architecture (e.g., UNet) and backbone (e.g., ResNet-34)
3. Under **Weight Initialization**, switch to advanced mode and select **Use SSL pretrained encoder**
4. Browse to the `model.pt` that SSL pretraining just created
5. Train as normal

The classifier should now work much better on the new images because:
- The encoder already knows what features matter for fly wings (from your original training)
- The encoder has adapted to how those features look in the new images (from SSL pretraining)
- The decoder learned to classify using the adapted features (from your few annotations)

## Tips

- **BYOL vs SimCLR**: Use BYOL when you have fewer than ~200 tiles. SimCLR works better with larger datasets but needs bigger batch sizes.
- **How many unannotated images?**: More is better, but even 10-20 images help. The goal is to give the encoder enough examples of the new image style.
- **How many annotated images for Step 3?**: Start with 3-5. If results aren't good enough, add a few more. You'll likely need far fewer annotations than training from scratch.
- **Same backbone matters**: The SSL pretrained encoder must use the same backbone as your original model. If your original used ResNet-34, the SSL pretraining must also use ResNet-34.
- **Check your model's backbone**: Open the `metadata.json` file next to your trained `model.pt` in any text editor. Look for `"architecture"` -> `"backbone"` to see which backbone was used.

---

## Other Situations Where SSL Pretraining Helps

The domain adaptation scenario above (different microscope, same biology) is the most common use case, but SSL pretraining is useful in several other situations. In each case, the idea is the same: let the encoder learn what your images look like before you ask it to classify them.

### You have lots of images but little time to annotate

**Situation:** You have 500 whole-slide images of fly wings, but labeling veins and hairs is tedious. You can only afford to annotate 5-10 of them thoroughly.

**What to do:**
1. Draw rough tissue outlines on all 500 images (fast -- just boxes or lasso around the wing)
2. Run SSL Pretrain on all 500 images (no source model needed -- train from scratch)
3. Annotate veins/hairs/background carefully on 5-10 images
4. Train using the SSL encoder

**Why it works:** The encoder has already seen 500 images worth of visual patterns. It knows what wing tissue looks like, what edges look like at your resolution, and what artifacts to ignore. It just needs a few labeled examples to learn which patterns correspond to which classes.

### You are adding a new class to an existing classifier

**Situation:** Your classifier detects veins and hairs, and now you need to add a "trichome" class. You have very few examples of trichomes annotated, but plenty of images that contain them.

**What to do:**
1. Run SSL Pretrain on images containing trichomes, using your existing trained model as the source
2. Add trichome annotations to a few images
3. Retrain with all three classes, using the SSL encoder as the starting point

**Why it works:** The encoder adapts to the visual characteristics of trichome-containing regions while preserving its existing knowledge of veins and hairs. The few trichome annotations are enough because the encoder already understands the tissue context.

### Your images have unusual properties

**Situation:** You are working with:
- Polarized light microscopy (not standard brightfield)
- Very high or very low magnification
- Non-standard staining or unstained specimens
- Compressed images (JPEG artifacts)
- Images with unusual bit depth or dynamic range

**What to do:**
1. Collect 50+ representative tiles from your images
2. Run SSL Pretrain from scratch (no source model) using the matching backbone
3. Train a supervised classifier using the SSL encoder

**Why it works:** ImageNet-pretrained encoders learned features from natural photographs -- dogs, cars, landscapes. Those features transfer surprisingly well to many microscopy images, but when your images are very different from photographs (polarized light, unusual stains), the transfer is poor. SSL pretraining on your actual images teaches the encoder what features matter for *your* data specifically.

### Multi-site or multi-scanner studies

**Situation:** You are running a study with images from 5 different sites. Each site uses a different scanner, and the color balance, resolution, and compression differ between sites. A classifier trained on Site A performs inconsistently on Sites B-E.

**What to do:**
1. Pool unannotated images from all 5 sites
2. Run SSL Pretrain on the pooled images (from scratch or from an existing model)
3. Annotate a few representative images from each site
4. Train a single classifier using the SSL encoder

**Why it works:** The SSL encoder learns features that are consistent across all 5 sites. It effectively normalizes the visual differences between scanners. The supervised classifier then works on any site because the encoder produces similar feature representations regardless of which scanner acquired the image.

### You want to improve a trained model without more annotations

**Situation:** Your classifier is decent but makes mistakes in certain regions. You don't want to annotate more, but you have additional unannotated images that cover those problem areas.

**What to do:**
1. Run SSL Pretrain on the additional images, using your existing model as the source
2. Retrain using the SSL encoder with your existing annotations (same annotations, better encoder)

**Why it works:** The encoder becomes better at representing the difficult regions because it has practiced on more examples of them. Even though the annotations haven't changed, the features the encoder extracts are better, so the decoder makes fewer mistakes.

---

## Choosing the Right Pretraining Approach

| Situation | Approach | Source model? | Method |
|-----------|----------|---------------|--------|
| Different microscope/scanner | Domain-adaptive SSL | Yes (existing model) | BYOL |
| Large unlabeled dataset, few labels | SSL from scratch | No | SimCLR |
| Adding a new class | Domain-adaptive SSL | Yes (existing model) | BYOL |
| Unusual imaging modality | SSL from scratch | No | BYOL |
| Multi-site study | SSL from scratch on pooled data | No | SimCLR |
| Improving without more labels | Domain-adaptive SSL | Yes (existing model) | BYOL |
| Standard H&E histopathology | Skip SSL -- use histology-pretrained backbone | N/A | N/A |

For more details on when to use each method (SimCLR vs BYOL), see [Best Practices: SSL Pretraining](BEST_PRACTICES.md#ssl-pretraining-simclr--byol).

For parameter descriptions, see [Parameters: SSL Pretraining](PARAMETERS.md#ssl-pretraining-parameters).
