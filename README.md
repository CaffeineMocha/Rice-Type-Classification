# 🍚 Rice Type Classification with ResNet50

This project implements a **deep learning pipeline** for classifying **5 rice varieties** using **transfer learning with ResNet50** in PyTorch.  
It includes **training, evaluation, and inference scripts**, as well as a **FastAPI deployment** for serving the model.

---

## 📂 Repository Structure

```
Rice-Classification-ResNet50/
│
├── src/
│   └── main.py                   # Training, validation, testing, inference
│
├── api/
│   └── rice_classifier_api.py    # FastAPI deployment script
│
├── models/
│   └── resnet50_rice_classifier.pth   # Trained weights (not included in repo)
│
├── notebooks/
│   └── Rice_Classification.ipynb # Jupyter Notebook workflow
│
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── .gitignore
```

---

## 📊 Dataset

- The project uses the **Rice Image Dataset**, containing 5 classes:
  - Arborio  
  - Basmati  
  - Ipsala  
  - Jasmine  
  - Karacadag  

⚠️ The dataset is **not included** in this repo.  
You must download it manually and place it under `data/` or update the dataset path in the code.

---

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Rice-Classification-ResNet50.git
   cd Rice-Classification-ResNet50
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Check GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

---

## 🏋️ Training & Evaluation

Run the training pipeline:

```bash
python src/main.py
```

The script will:
- Train ResNet50 with transfer learning
- Evaluate on validation/test sets
- Save the best model to `models/resnet50_rice_classifier.pth`
- Generate accuracy/loss curves
- Print classification report and confusion matrix

---

## 📈 Results

- Best Validation Accuracy: **XX%**
- Final Test Accuracy: **XX%**
- Macro F1 Score: **XX%**

Example confusion matrix:

![Confusion Matrix](docs/confusion_matrix.png)

---

## 🔮 Inference (from Notebook or Script)

Use the `predict_image` function to classify a new image:

```python
from src.main import predict_image
from PIL import Image
import io

with open("example.jpg", "rb") as f:
    image_bytes = f.read()

predicted_class, confidence = predict_image(image_bytes, model, idx_to_class, device)
print(predicted_class, confidence)
```

---

## 🌐 Deployment with FastAPI

We provide a **FastAPI app** for model serving (`rice_classifier_api.py`).

### Run the API locally

```bash
uvicorn api.rice_classifier_api:app --reload
```

- API will be available at: `http://127.0.0.1:8000/`
- Swagger docs at: `http://127.0.0.1:8000/docs`

### Endpoints
- `GET /` → Welcome message  
- `GET /health` → Health check  
- `POST /predict` → Upload an image (`jpg/png/bmp/webp`) and get prediction  

Example response:

```
Jasmine (97.53%)
```

---

## 🛠️ Technologies Used
- Python 3.9+
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- NumPy, Matplotlib, Seaborn, scikit-learn
- PIL (Pillow)

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgments
- [Rice Image Dataset](https://www.muratkoklu.com/datasets/)  
- PyTorch team for ResNet50 pretrained weights  
