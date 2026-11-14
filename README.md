# PixelTrace

An image classifier that predicts whether a picture is AI-generated or real.

<!-- Project image goes here. -->

## Project details

- **Dataset:** [Parveshiiii/AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real)
- **Dataset size:** 13,999 images
- **Model:** VGG16 pretrained CNN
- **Framework:** TensorFlow / Keras
- **Task:** Binary image classification
- **Labels:** `0` = AI-generated, `1` = Real
- **Split:** 75% train, 12.5% validation, 12.5% test

## Files

```text
Pixeltrace/
├── train.py
├── predict.py
├── inspect_dataset.py
├── requirements.txt
├── model/
│   └── Ai_Or_Not.keras
└── samples/
    ├── ai_01.png
    ├── ai_02.png
    ├── ai_03.png
    ├── real_01.jpg
    ├── real_02.jpg
    └── uncertain_01.png
```

```text
Input image
    ↓
VGG16 (frozen feature extractor)
    ↓
Dense (256, ReLU)
    ↓
Dense (1, Sigmoid)
    ↓
AI-generated / Real
```

## Install

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run prediction

```bash
python predict.py
```

The script loads `model/Ai_Or_Not.keras` and prints the raw score plus `AI` or `REAL` for each image in `samples/`.

## Train

```bash
python train.py
```

This downloads the dataset, creates the 75% / 12.5% / 12.5% split, trains the model, evaluates it, and saves the result to `model/Ai_Or_Not.keras`.

## Inspect the dataset

```bash
python inspect_dataset.py
```

The dataset is downloaded from [Hugging Face](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real).

