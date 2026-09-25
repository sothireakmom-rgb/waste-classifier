# Waste Classification for Recycling

A deep learning system that classifies images of waste into six categories — **cardboard, glass, metal, paper, plastic, trash** — to support automated recycling sorting.

## Overview

**What it does.** Given a photo of a waste item, the model predicts which of the six material categories it belongs to and returns a confidence score for each.

**Why it matters.** Sorting waste by hand is slow and error-prone, and the mistakes are expensive: a single misplaced item can contaminate an entire batch of otherwise recyclable material and send it to landfill. Automating the classification step lets recyclable materials be sorted more accurately and efficiently — which matters both for cutting processing costs and for reducing contamination in recycling streams.

**How it works.** Three deep learning architectures were built and compared — a CNN trained from scratch, a frozen pretrained ResNet-18, and a fully fine-tuned ResNet-18 — to study how much pretrained knowledge and fine-tuning improve performance. The fine-tuned ResNet-18 achieved the best result at **92.1% test accuracy**.

## Dataset

[TrashNet](https://huggingface.co/datasets/garythung/trashnet) (`garythung/trashnet` on Hugging Face) — 5,054 labeled images of waste items photographed against plain backgrounds.

| Split | Images |
|---|---|
| Train | 3,638 |
| Validation | 405 |
| Test | 1,011 |
| **Total** | **5,054** |

| Category | Images |
|---|---|
| Cardboard | 806 |
| Glass | 1,002 |
| Metal | 820 |
| Paper | 1,188 |
| Plastic | 964 |
| Trash | 274 |

The `trash` category (miscellaneous non-recyclable items) is noticeably underrepresented — roughly a quarter the size of `paper`.

## Method

The project is built in **PyTorch / torchvision**. Three approaches were trained on the same splits and evaluated on the same held-out test set, so the numbers are directly comparable.

**Approach 1 — CNN from scratch.** A convolutional network trained from random initialization, with no pretrained weights. This is the baseline: it shows what the dataset alone can support.

**Approach 2 — ResNet-18, frozen backbone.** ResNet-18 pretrained on ImageNet with the convolutional backbone frozen; only a new 6-class classifier head is trained. The pretrained features are reused as-is, which is fast to train and hard to overfit.

**Approach 3 — ResNet-18, fine-tuned.** The same pretrained ResNet-18, but the deeper layers are unfrozen and updated along with the classifier head, letting the network adapt its features to waste imagery instead of relying on generic ImageNet features. This performed best and is the model served by the demo app.

All approaches use ImageNet normalization on 224×224 inputs:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## Results

| Approach | Test accuracy |
|---|---|
| 1. CNN from scratch | 75.6% |
| 2. ResNet-18 (frozen backbone) | 84.2% |
| 3. ResNet-18 (fine-tuned) | **92.1%** |

Transfer learning accounts for most of the gain: moving from a scratch-trained CNN to frozen ImageNet features adds ~8.6 points, and unfreezing the backbone for fine-tuning adds a further ~7.9 points on top of that.

Per-approach learning curves, confusion matrices, and the raw comparison table:

| File | Contents |
|---|---|
| `results_comparison.csv` | Accuracy and metrics for all three approaches |
| `chart_accuracy_comparison.png` | Test accuracy across the three approaches |
| `chart_learning_curves.png` | Train/validation curves per approach |
| `chart_confusion_matrices.png` | Confusion matrices per approach |

These files are included directly in the [`results/`](results/) folder of this repo — no external download needed.

## Real-world testing

The classifier was tested on photos taken outside the dataset to see how well it generalizes beyond TrashNet's plain-background studio images.

> These observations come from the **earlier TensorFlow/MobileNetV2 version** of the project (87.9% test accuracy) and have not yet been re-run against the fine-tuned ResNet-18. The qualitative pattern is expected to hold, but the confidence figures are specific to the old model.

| Test image | Predicted | Confidence | Correct? | Notes |
|---|---|---|---|---|
| Plastic bottle, clean background | Plastic | 91.3% | ✅ | Single, clearly visible item — the case the model was trained for |
| Landfill pile (mixed waste) | Metal | 79.4% | ❌ | Cluttered multi-material scene; high confidence in a wrong answer |
| Crushed can pile | Plastic | 82.3% | ❌ | Many overlapping objects confuse a single-label classifier |
| Plastic bag in grass | Glass | 44.4% | ❌ | Camouflaged against a busy background; low confidence reflects the uncertainty |

The pattern is clear: the model is reliable on single items against plain backgrounds and unreliable on anything else.

## Limitations

- **Single-item assumption.** The model outputs one label per image, but real waste arrives in piles. On cluttered, multi-object scenes it picks one material and often reports high confidence while being wrong — a failure mode that is worse than an obvious error because it looks trustworthy.
- **Needs object detection for real scenes.** Handling multi-item images properly requires detecting and localizing each object first, e.g. with YOLO, then classifying each detection.
- **Background sensitivity.** Camouflaged items against busy backgrounds degrade badly — the training images are all shot on plain, uniform backgrounds.
- **Class imbalance.** The `trash` category has only 274 training images, far fewer than the others, so performance on genuinely non-recyclable items is the least trustworthy.
- **Dataset size.** 5,054 images is small for training a CNN from scratch, which is much of why Approach 1 lags so far behind the transfer-learning approaches.

## Demo app

[`app.py`](app.py) serves the fine-tuned ResNet-18 through a Gradio interface: upload a photo, get probabilities across all six categories.

```bash
pip install -r requirements.txt
python app.py
```

The app expects `approach3_resnet_finetuned.pth` in the project root. If it is missing, the app exits with a message telling you to download it.

## Model weights

Trained weight files are **not committed to this repo** — they are hosted on Google Drive instead.

| File | Approach | Download |
|---|---|---|
| `approach1_cnn_scratch.pth` | CNN from scratch | [Download](https://drive.google.com/file/d/19kDdEU3QqTYRWPXUxMCbt4jOgjMjN7Rj/view?usp=sharing) |
| `approach2_resnet_frozen.pth` | ResNet-18, frozen backbone | [Download](https://drive.google.com/file/d/10TRne6743a9o2oNUHA5rCrJ_VxPtpxvG/view?usp=sharing) |
| `approach3_resnet_finetuned.pth` | ResNet-18, fine-tuned | [Download](https://drive.google.com/file/d/1BI2ZuO9LgARR0lG4iZ4GwrNmnmLTuZWq/view?usp=sharing) |

Place the downloaded `.pth` files in the project root.

## Project structure

```
waste-classifier/
├── app.py                             # Gradio demo app — serves the fine-tuned ResNet-18
├── requirements.txt                   # Python dependencies (PyTorch stack)
├── README.md                          # This file
├── DL.ipynb                           # Legacy TensorFlow/Keras notebook, superseded by notebooks/DLW.ipynb
├── approach3_resnet_finetuned.pth     # Fine-tuned ResNet-18 weights — not committed, download from Drive
├── notebooks/
│   └── DLW.ipynb                      # Main PyTorch notebook: data loading, all three approaches, evaluation
├── results/
│   ├── results_comparison.csv         # Trainable params, training time, and test accuracy per approach
│   ├── chart_accuracy_comparison.png  # Test accuracy across the three approaches
│   ├── chart_learning_curves.png      # Train/validation curves per approach
│   └── chart_confusion_matrices.png   # Confusion matrices per approach
└── slides/                            # Presentation materials (currently empty)
```

The three `.pth` weight files are not tracked in git — see [Model weights](#model-weights) for the download links.

## Tools used

- **PyTorch / torchvision** — model definition, training, and the pretrained ResNet-18
- **Hugging Face Datasets** — dataset loading
- **scikit-learn** — evaluation metrics and confusion matrices
- **matplotlib / seaborn / pandas** — results analysis and charts
- **Gradio** — demo interface
- **Google Colab (T4 GPU)** — model training
- **VS Code** — project organization

## Future work

- Collect or augment more data for the underrepresented `trash` category.
- Re-run the real-world testing against the fine-tuned ResNet-18 to replace the older figures above.
- Explore YOLO for multi-item detection so cluttered scenes can be handled object by object rather than as a single label.
- Train longer and experiment with learning-rate schedules and stronger augmentation on the fine-tuned model.

## AI use

Claude (Claude Code) was used to help debug PyTorch code, fix dataset loading issues, and restructure the project. All analysis and results in this repository represent my own work and were verified manually.
