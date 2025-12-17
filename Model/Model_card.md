\# Model Card: Stain Classification Model (YOLOv8-cls)

\## Model Overview

This model is a convolutional neural network trained to classify different types of stains

(e.g., coffee, wine, tomato sauce, ink, chocolate) from images of fabric or textile surfaces.

The model is intended for educational and research purposes as part of a university capstone project.

- \*\*Model type\*\*: Image classification (YOLOv8-cls)
- \*\*Framework\*\*: PyTorch (Ultralytics YOLOv8)
- \*\*Input\*\*: RGB image
- \*\*Output\*\*: Stain class label with confidence score

\---

\## Training Data

The model was trained on a custom dataset consisting of images of fabric stains.

The dataset includes multiple stain categories captured under varying lighting conditions

and backgrounds.

- \*\*Classes\*\*: coffee, wine, tomato\_sauce, ink, chocolate, clean, blood, juice, dirt\_mud
- \*\*Dataset size\*\*: [~1200 images]
- \*\*Train/Validation split\*\*: [80% / 20%]
- \*\*Data source\*\*: Web-collected and curated for academic use
- \*\*Preprocessing\*\*:
- Image resizing to model input resolution
- Normalization
- Basic data augmentation (random flip, rotation)

\---

\## Training Procedure

- \*\*Base model\*\*: YOLOv8 classification backbone
- \*\*Optimizer\*\*: [Adam (default setting in YOLOv8)]
- \*\*Loss function\*\*: Cross-entropy loss
- \*\*Number of epochs\*\*: [30]
- \*\*Batch size\*\*: [e.g., 32]
- \*\*Hardware\*\*: [CPU]

\---

\## Evaluation Results

Performance was evaluated on a held-out validation set.

- \*\*Validation accuracy\*\*: [84.81%]
- \*\*Best checkpoint\*\*: `best.pt`
- \*\*Selection criterion\*\*: Highest validation accuracy

Detailed training metrics are provided in `training\_log.txt`.

\---

\## Known Limitations

- The model may struggle with very small or faint stains.
- Visually similar stains (e.g., dark coffee vs. chocolate) are occasionally misclassified.
- Performance may degrade under extreme lighting or unusual fabric textures.

\---

\## Intended Use

- Academic demonstrations and coursework
- Prototype stain classification systems
- Educational exploration of computer vision models

\---

\## Not Intended Use

- Medical or safety-critical applications
- Commercial deployment without further validation
- Automated decision-making in real-world laundry services

\---

\## Ethical Considerations

This model does not process personal or sensitive data.

All images were collected and used for academic purposes only.

\---

\## How to Use the Model

1. Download or locate the trained model file `best.pt` in the `model/` directory.
1. Run the application using:

\```bash

python src/app.py
