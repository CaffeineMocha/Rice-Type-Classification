from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont
import io, os, base64
import torch
import torchvision.transforms as transforms
from torchvision import models

MODEL_WEIGHTS = "resnet50_rice_classifier.pth"
NUM_CLASSES   = 5
FALLBACK_IDX_TO_CLASS = {
    0: "Arborio",
    1: "Basmati",
    2: "Ipsala",
    3: "Jasmine",
    4: "Karacadag",
}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

ckpt = torch.load(MODEL_WEIGHTS, map_location=device)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("[load] missing keys:", missing)
    print("[load] unexpected keys:", unexpected)
    idx_to_class = ckpt.get("idx_to_class", FALLBACK_IDX_TO_CLASS)
else:
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("[load] missing keys:", missing)
    print("[load] unexpected keys:", unexpected)
    idx_to_class = FALLBACK_IDX_TO_CLASS

model.to(device)
model.eval()

app = FastAPI(title="Rice Classifier API")

@app.get("/")
def home():
    return {"message": "Rice Classifier API. Open /docs to try the /predict endpoint."}

def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = TEST_VAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    predicted_class = idx_to_class[pred.item()]
    confidence = float(conf.item()) * 100.0
    return predicted_class, confidence

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.responses import PlainTextResponse

@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return "Invalid file type"

    image_bytes = await file.read()
    predicted_class, confidence = predict_image(image_bytes)
    return f"{predicted_class} ({confidence:.2f}%)"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
