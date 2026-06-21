# Pipeline de inferencia — Detección de acné facial (4 modelos)

Paquete autónomo para integrar **los cuatro modelos** en el aplicativo web. **No** depende del
código de entrenamiento ni del notebook.

## Archivos
| Archivo | Qué es |
|---|---|
| `inference.py` | Pipeline completo: **segmentación del rostro** + preprocesamiento + predicción con los 4 modelos. |
| `modelos/` | Los 4 artefactos: `vit.pt`, `mobilenetv2.pt`, `resnet50.pt`, `efficientnetv2s.pt`. |
| `face_detection_yunet_2023mar.onnx` | Modelo de detección de rostro (YuNet, OpenCV). **Ya incluido.** |
| `requirements.txt` | Dependencias mínimas. |
| `app_fastapi.py` | Ejemplo de API REST (opcional). |

## Instalación
```bash
pip install -r requirements.txt
```

## Uso (3 líneas)
```python
from inference import MultiAcnePredictor
predictor = MultiAcnePredictor("modelos")     # carga los 4 UNA sola vez al arrancar
predictor.predict("selfie.jpg")               # los 4 modelos + consenso
predictor.predict("selfie.jpg", modelo="vit") # un solo modelo
```

## Salida — todos los modelos
```json
{
  "face_detected": true,
  "metodo_recorte": "oval_opencv",
  "modelos": {
    "vit": {"label": "acne", "prob_acne": 0.84},
    "mobilenetv2": {"label": "acne", "prob_acne": 0.99},
    "resnet50": {"label": "acne", "prob_acne": 0.99},
    "efficientnetv2s": {"label": "acne", "prob_acne": 0.99}
  },
  "consenso": {"label": "acne", "prob_acne_promedio": 0.95, "votos_acne": 4, "total_modelos": 4}
}
```

## Mejora clave (v2): segmentación del rostro
El problema de las primeras pruebas (el modelo decía "sin acné" en fotos con acné) era un **gap de
dominio**: los modelos se entrenaron con **rostros segmentados en un óvalo sobre fondo negro**, no con
selfies crudas. Antes se hacía un **recorte rectangular** que incluía pelo, cuello y fondo —algo que el
modelo nunca vio—.

Ahora el pipeline **segmenta el rostro en un óvalo sobre fondo negro**, imitando el dataset:
1. **Detección de rostro con YuNet** (detector DNN de OpenCV, robusto a frontal, perfil y ángulos).
   Si por algún motivo no estuviera el `.onnx`, usa de respaldo las cascadas **Haar** (frontal + perfil).
2. Se aplica una **máscara ovalada** y se centra el rostro sobre un **lienzo negro** cuadrado.
3. Sobre esa imagen se aplica el **mismo preprocesamiento** del entrenamiento (resize 224, CLAHE en
   canal Y, normalización ImageNet) y se predice con los 4 modelos.

El campo `metodo_recorte` indica qué se usó: `oval_opencv` (rostro segmentado) o `imagen_completa`
(no se detectó rostro → conviene pedir una selfie frontal y bien iluminada).

> **Nota honesta:** la segmentación cierra la brecha de *encuadre/fondo*, pero **no** elimina otras dos
> diferencias entre el dataset y las selfies reales: la **severidad** (el dataset etiqueta acné graduado)
> y las condiciones de **piel/iluminación/cámara**. Por eso, en acné muy leve o en pieles/luces muy
> distintas, la predicción puede seguir siendo conservadora.

## Notas de integración
- **Carga el predictor una sola vez** al iniciar el servidor (no por request).
- Funciona en **CPU**; con GPU es más rápido pero no obligatorio.
- **No descargues nada:** el modelo de detección de rostro ya viene en la carpeta.
- Tamaño de los `.pt`: ViT ≈ 87 MB, ResNet-50 ≈ 94 MB, EfficientNetV2-S ≈ 82 MB, MobileNetV2 ≈ 9 MB.
- Los `.pt` incluidos son una muestra; reemplázalos por los exportados desde Colab (celda
  "Exportar los 4 modelos para la web" → `modelos_web.zip`).
