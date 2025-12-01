# Test Backend Server For Facial Recignition

## 1️⃣ Create virtual environment
python -m venv .venv

## 2️⃣ Activate venv
& .\.venv\Scripts\Activate.ps1

## 3️⃣ Install dependencies
pip install -r requirements.txt

## 4️⃣ Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## 5️⃣ (Optional) expose to internet
ngrok http 8000
