# ChestX-PneumoDetect: Pneumonia Detection from X-rays

## 🩺 Problem Description
Pneumonia is a leading cause of death worldwide. Early diagnosis via chest X-ray can be life-saving. This project builds a deep learning model to automatically classify chest X-ray images as **NORMAL** or **PNEUMONIA** using **PyTorch** (ResNet18) and **TensorFlow/Keras** (MobileNetV2).

The solution is deployed as a FastAPI web service, capable of running in a Docker container.

## 📂 Project Structure
```
├── data/               # Dataset storage
├── models/             # Trained models (model_pth.pth, model_keras.h5)
├── notebook.ipynb      # Exploratory Data Analysis & Training experiments
├── train.py            # Script to train models
├── predict.py          # FastAPI inference application
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── download_data.sh    # Dataset download script
```

## 🚀 How to Run

### 1. Prerequisites
- **Python 3.10+**
- **Docker** (optional, for containerization)
- **Kaggle API Token** (for downloading dataset)

### 2. Clone & Setup
Clone the repository and enter the directory:
```bash
git clone https://github.com/code-hy/chestxray-pneumonia-detection.git
cd chestxray-pneumonia-detection
```

### 3. Environment Setup
Create a virtual environment and install dependencies:

**Option A: Using standard pip (Recommended)**
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using uv**
```bash
uv venv
uv sync
```

### 4. Download Dataset
You need a Kaggle API token (`kaggle.json`).
1.  Place `kaggle.json` in `~/.kaggle/` (user home directory).
2.  Run the download script:
```bash
# Windows (Git Bash) or Linux/Mac
bash download_data.sh
```

### 5. Training (Optional)
**⚠️ Note:** Training can take 2-3 hours depending on your hardware. 
Pre-trained models are already provided in the `models/` directory.

To retrain the models:
```bash
python train.py
```
This will:
- Train a PyTorch ResNet18 model.
- Train a Keras MobileNetV2 model.
- Save artifacts to `models/`.

### 6. Run Prediction API (FastAPI)
Start the local server:
```bash
uvicorn predict:app --host 0.0.0.0 --port 5000 --reload
```
- **Swagger UI**: Visit [http://localhost:5000/docs](http://localhost:5000/docs) to test the API interactively.
- **Prediction Endpoint**: `POST /predict` (Upload an X-ray image).

---

### 7. Run using test_form.html (Optional)

Open `test_form.html` in a web browser to test the API interactively.
and then upload an X-ray image to test the API.

---
## 🐳 Docker Deployment

### 1. Build the Docker Image
Ensure you are in the project root (where `Dockerfile` is located):
```bash
docker build -t chestx-app .
```

### 2. Run the Container
```bash
docker run -p 5000:5000 chestx-app
```
The API is now available at `http://localhost:5000`.

---

## ☁️ Deploy to Render

1.  **Push to GitHub**: Ensure your code is in a GitHub repository.
2.  **Create New Web Service** on [Render](https://render.com/).
3.  **Connect GitHub**: Select your repository.
4.  **Configure**:
    - **Runtime**: Docker
    - **Region**: (Select closest to you)
    - **Branch**: main
5.  **Deploy**: Render will automatically build the Docker image and start the service.

---

## 📊 Results and Evaluation
See `notebook.ipynb` for detailed confusion matrices and performance metrics. The PyTorch ResNet18 model typically achieves >90% recall on Pneumonia cases.

### Screenshots of Predictions using test_form.html

<img width="695" height="636" alt="image" src="https://github.com/user-attachments/assets/1ee8b9e8-34f5-479a-8016-daf79352c0a8" />

<img width="638" height="684" alt="image" src="https://github.com/user-attachments/assets/bd8a3d10-687a-4346-8f8f-bee32bfec11a" />
