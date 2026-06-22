"""Pipeline de inferencia para el aplicativo web de detección de acné — 4 MODELOS.

Mejora clave (v2): el rostro se **SEGMENTA en un óvalo sobre fondo negro** para imitar el
dominio del dataset de entrenamiento (rostros segmentados, no recortes rectangulares).
  - Si MediaPipe está disponible -> segmentación facial precisa (malla de 468 puntos).
  - Si no -> respaldo con OpenCV (detección Haar + máscara elíptica). Siempre funciona.
  - Si no se detecta rostro -> se usa la imagen completa (y se avisa).

Dependencias mínimas: torch, torchvision, timm, opencv-python, numpy.
Opcional (recomendado): mediapipe.

Uso:
    from inference import MultiAcnePredictor
    pred = MultiAcnePredictor("modelos")
    pred.predict("selfie.jpg")                 # los 4 modelos + consenso
    pred.predict("selfie.jpg", modelo="vit")   # un solo modelo
"""
import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as tvm

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Detección de rostro 100% OpenCV (sin pip extra).
#   Principal: YuNet (DNN, robusto a ángulos/perfil) si está el modelo .onnx en la carpeta.
#   Respaldo: cascadas Haar (frontal + perfil).
_DIR = os.path.dirname(os.path.abspath(__file__))
_YUNET_PATH = os.path.join(_DIR, "face_detection_yunet_2023mar.onnx")
try:
    _YUNET = cv2.FaceDetectorYN.create(_YUNET_PATH, "", (320, 320), score_threshold=0.6) \
        if os.path.exists(_YUNET_PATH) else None
except Exception:
    _YUNET = None
_CASCADE_F = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CASCADE_P = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


# ----------------------------------------------------------------------------
# 1) Arquitecturas (sin pesos ImageNet: cargamos los nuestros)
# ----------------------------------------------------------------------------
def build_model(name: str) -> nn.Module:
    if name == "mobilenetv2":
        m = tvm.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == "resnet50":
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "efficientnetv2s":
        m = tvm.efficientnet_v2_s(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif name == "vit":
        m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=1)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return m


# ----------------------------------------------------------------------------
# 2) Segmentación del rostro en óvalo sobre fondo negro (imita el dataset)
# ----------------------------------------------------------------------------
def _square_on_black(masked, cx, cy, side, dtype):
    canvas = np.zeros((side, side, 3), dtype=dtype)
    x0, y0 = cx - side // 2, cy - side // 2
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, masked.shape[1]), min(y0 + side, masked.shape[0])
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = masked[sy0:sy1, sx0:sx1]
    return canvas


def _detect_yunet(img_rgb):
    if _YUNET is None:
        return None
    h, w = img_rgb.shape[:2]
    _YUNET.setInputSize((w, h))
    _, faces = _YUNET.detect(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))  # YuNet usa BGR
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda r: r[2] * r[3])
    return int(f[0]), int(f[1]), int(f[2]), int(f[3])


def _detect_haar(gray):
    faces = list(_CASCADE_F.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)))
    if not faces:
        faces = list(_CASCADE_P.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)))
    if not faces:
        W = gray.shape[1]
        flipped = _CASCADE_P.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=(60, 60))
        faces = [(W - x - w, y, w, h) for (x, y, w, h) in flipped]
    return max(faces, key=lambda f: f[2] * f[3]) if faces else None


def _detect_face(img_rgb):
    f = _detect_yunet(img_rgb)
    if f is None:
        f = _detect_haar(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    return f


def segment_face(img_rgb, margin=0.18):
    """Segmenta el rostro en un ÓVALO sobre fondo negro (imita el dataset).
    Devuelve (imagen_segmentada, rostro_detectado, metodo)."""
    f = _detect_face(img_rgb)
    if f is None:
        return img_rgb, False, "imagen_completa"
    x, y, w, h = f
    cx, cy = x + w // 2, y + h // 2
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    # elipse que aproxima el óvalo del rostro (un poco más alta que ancha)
    cv2.ellipse(mask, (cx, cy), (int(w * 0.62), int(h * 0.78)), 0, 0, 360, 255, -1)
    masked = np.zeros_like(img_rgb); masked[mask > 0] = img_rgb[mask > 0]
    side = int(max(w, h) * (1 + margin))
    return _square_on_black(masked, cx, cy, side, img_rgb.dtype), True, "oval_opencv"


# ----------------------------------------------------------------------------
# 3) Preprocesamiento idéntico al de validación del entrenamiento
# ----------------------------------------------------------------------------
def preprocess(img_rgb, img_size, mean, std) -> torch.Tensor:
    img = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def _read_rgb(image):
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(image)


# ----------------------------------------------------------------------------
# 4) Predictor de un solo modelo
# ----------------------------------------------------------------------------
class AcnePredictor:
    def __init__(self, weights_path, device=None):
        self.weights_path = weights_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Evitamos llamar a torch.load al arrancar para no consumir memoria RAM (evita error OOM 512MB)
        filename = os.path.basename(weights_path)
        self.model_name = filename.replace(".pt", "")
        self.class_names = ['no_acne', 'acne']
        self.img_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.threshold = 0.5

    @torch.no_grad()
    def prob_from_tensor(self, x):
        import gc
        # Cargar pesos del modelo pesados en memoria únicamente durante la inferencia
        ckpt = torch.load(self.weights_path, map_location=self.device)
        model = build_model(self.model_name).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        prob = torch.sigmoid(model(x.to(self.device))).item()
        
        # Eliminar el modelo e invocar al recolector de basura de Python para liberar RAM al instante
        del model
        del ckpt
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        gc.collect()
        
        return prob

    def predict(self, image, do_face_crop=True):
        img = _read_rgb(image); face = None; method = "ninguno"
        if do_face_crop:
            img, face, method = segment_face(img)
        x = preprocess(img, self.img_size, self.mean, self.std)
        p = self.prob_from_tensor(x)
        return {"model": self.model_name, "label": self.class_names[int(p >= self.threshold)],
                "prob_acne": round(p, 4), "threshold": self.threshold,
                "face_detected": face, "metodo_recorte": method}


# ----------------------------------------------------------------------------
# 5) Predictor de los CUATRO modelos
# ----------------------------------------------------------------------------
class MultiAcnePredictor:
    ORDEN = ["vit", "mobilenetv2", "resnet50", "efficientnetv2s"]

    def __init__(self, modelos_dir="modelos", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictors = {}
        for f in sorted(os.listdir(modelos_dir)):
            if f.endswith(".pt"):
                p = AcnePredictor(os.path.join(modelos_dir, f), device=self.device)
                self.predictors[p.model_name] = p
        if not self.predictors:
            raise RuntimeError(f"No se encontraron modelos (.pt) en {modelos_dir}")
        ref = next(iter(self.predictors.values()))
        self.img_size, self.mean, self.std = ref.img_size, ref.mean, ref.std
        self.class_names = ref.class_names

    def modelos(self):
        return list(self.predictors.keys())

    def predict(self, image, modelo=None, do_face_crop=True):
        img = _read_rgb(image); face = None; method = "ninguno"
        if do_face_crop:
            img, face, method = segment_face(img)      # se segmenta UNA sola vez
        x = preprocess(img, self.img_size, self.mean, self.std)

        if modelo is not None:
            p = self.predictors[modelo].prob_from_tensor(x)
            return {"modelo": modelo, "label": self.class_names[int(p >= 0.5)],
                    "prob_acne": round(p, 4), "face_detected": face, "metodo_recorte": method}

        res = {}
        for name in self.ORDEN:
            if name in self.predictors:
                p = self.predictors[name].prob_from_tensor(x)
                res[name] = {"label": self.class_names[int(p >= 0.5)], "prob_acne": round(p, 4)}
        probs = [r["prob_acne"] for r in res.values()]
        votos = sum(1 for r in res.values() if r["label"] == "acne")
        prom = float(np.mean(probs))
        consenso = {"label": self.class_names[int(prom >= 0.5)], "prob_acne_promedio": round(prom, 4),
                    "votos_acne": votos, "total_modelos": len(res)}
        return {"face_detected": face, "metodo_recorte": method, "modelos": res, "consenso": consenso}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="modelos")
    ap.add_argument("--image", required=True)
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--no-face-crop", action="store_true")
    a = ap.parse_args()
    pred = MultiAcnePredictor(a.modelos)
    print("Modelos:", pred.modelos())
    print(json.dumps(pred.predict(a.image, modelo=a.modelo, do_face_crop=not a.no_face_crop),
                     ensure_ascii=False, indent=2))
