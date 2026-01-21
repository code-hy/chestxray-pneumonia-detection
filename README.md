# ChestX-PneumoDetect: Pneumonia Detection from X-rays

## 🩺 Problem Description
Pneumonia is a leading cause of death worldwide, especially in children and the elderly. Early diagnosis via chest X-ray can be life-saving, but radiologists are scarce in many regions. This project builds a deep learning model to automatically classify chest X-ray images as **NORMAL** or **PNEUMONIA** using convolutional neural networks (CNNs) and transfer learning.

The solution can be deployed as a lightweight web service for clinics or mobile apps to assist preliminary screening.

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- `uv` installed: https://docs.astral.sh/uv/

### 1. Clone & Setup
```bash
git clone https://github.com/yourname/chestx-pneumodetect.git
cd chestx-pneumodetect

### 2. Install Dependencies (using uv)
```bash
uv sync --all-extras
```
This creates a virtual environment and installs all packages from `pyproject.toml`

### 3. Download Dataset
```bash
chmod +x data/download_data.sh
./data/download_data.sh
```
This downloads the ChestX dataset from the NIH ChestX dataset and places it in the `data` directory.

### 4. Explore and Train
* Open ##notebook.ipynb for EDA and model comparison.
* Train final model:
```bash
uv run -m train.py
```
