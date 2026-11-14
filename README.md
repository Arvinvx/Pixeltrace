# PixelTrace

An image classifier that predicts whether a picture is AI-generated or real.

<!-- Project image goes here. -->

| Component | Details |
| --- | --- |
| Dataset | [Parveshiiii/AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) |
| Dataset size | 13,999 images |
| Model | VGG16 pretrained CNN |
| Framework | TensorFlow / Keras |
| Task | Binary image classification |
| Labels | `0` = AI-generated, `1` = Real |
| Split | 75% train, 12.5% validation, 12.5% test |

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

