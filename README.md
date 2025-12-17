# [Stain Away - Image stain detector]
## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `python src/app.py`
3. Open browser: `http://127.0.0.1:7860`

See `docs/deployment_notes.md` for deployment details.

## Live Demo
https://huggingface.co/spaces/Genuine123459/Stain_detector

https://x.thunkable.com/copy/460390ㅡ6dfc3598fb3f1437f1348d00f6


## Dataset
Classes:
coffee, wine, tomato_sauce, ink, chocolate, blood, juice, dirt_mud, clean

Sources:
Images were collected from publicly available web sources and manually curated
for academic use.

Size:
Approximately 1200 images across 9 classes

Split:
Training set (80%), Validation set (20%)

License:
Mixed licenses depending on source.
The dataset is intended strictly for non-commercial, academic use.
See data/data_sheet.pdf for detailed documentation.

## Model
- **Tool**: Yolo v8n-cls
- **Accuracy**:

Epoch 1: ~36%

Epoch 15: ~80%

Epoch 30: ~84–85%
- **Export**:
Gradio + Python (app.py)

## Credits
Team:
ACDT G27
Junwoo Lee, Yongjun Lee, Ajung Kim, Yehseul Shin, Yeongchan Ju


Data:
Public web image sources (academic use only)

Tools & Libraries:
Python
PyTorch
Ultralytics YOLOv8
Gradio
Hugging Face Spaces

License

Code: MIT License

Model: Academic and research use only

Dataset: See data/data_sheet.pdf for license and usage details
