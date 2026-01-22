from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
from PIL import Image
import torch
import torchvision.transforms as T
import io
import os

app = FastAPI(title="ChestX-PneumoDetect API", description="Pneumonia detection from chest X-rays")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model_path = "models/model_pth.pth"

if not os.path.exists(model_path):
    raise RuntimeError("Model not found! Run `python train.py` first.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {"message": "ChestX-PneumoDetect API. Visit /docs for interactive UI."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = torch.softmax(model(img_t), dim=1)[0]
        
        return JSONResponse({
            "filename": file.filename,
            "prediction": {
                "normal": float(output[0]),
                "pneumonia": float(output[1])
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)