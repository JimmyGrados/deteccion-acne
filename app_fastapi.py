"""Ejemplo de endpoint para el aplicativo web con los CUATRO modelos.

Ejecutar:
    pip install fastapi uvicorn python-multipart
    uvicorn app_fastapi:app --reload --port 8000

Endpoints:
    GET  /health                 -> estado y lista de modelos
    POST /predict                -> los 4 modelos + consenso
    POST /predict?modelo=vit     -> solo un modelo (vit|mobilenetv2|resnet50|efficientnetv2s)

Probar:
    curl -F "file=@selfie.jpg" "http://localhost:8000/predict"
    curl -F "file=@selfie.jpg" "http://localhost:8000/predict?modelo=vit"
"""
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse

from inference import MultiAcnePredictor

app = FastAPI(title="Detección de acné facial — 4 modelos")
predictor = MultiAcnePredictor("modelos")   # carga los 4 una sola vez al arrancar


@app.get("/health")
def health():
    return {"status": "ok", "modelos": predictor.modelos()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), modelo: str = Query(None)):
    data = await file.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "imagen no válida"}
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = predictor.predict(img_rgb, modelo=modelo, do_face_crop=True)
    if result.get("face_detected") is False:
        result["aviso"] = "No se detectó un rostro; se usó la imagen completa. Pide una selfie frontal."
    return result


# ----------------------------------------------------------------------------
# Servir archivos estáticos del Frontend
# ----------------------------------------------------------------------------
@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.get("/styles.css")
def read_styles():
    return FileResponse("styles.css")


@app.get("/app.js")
def read_js():
    return FileResponse("app.js")


@app.get("/logo_esan.png")
def read_logo():
    return FileResponse("logo_esan.png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="127.0.0.1", port=8000, reload=True)
