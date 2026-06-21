document.addEventListener('DOMContentLoaded', () => {
    // Referencias de elementos DOM
    const uploadView = document.getElementById('upload-view');
    const modelSelectView = document.getElementById('model-select-view');
    const processingView = document.getElementById('processing-view');
    const resultView = document.getElementById('result-view');
    
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    
    const imagePreview = document.getElementById('image-preview');
    const resultImage = document.getElementById('result-image');
    
    const progressFill = document.getElementById('progress-fill');
    const pipelineStatus = document.getElementById('pipeline-status');
    
    const resultTitle = document.getElementById('result-title');
    const resultDesc = document.getElementById('result-desc');
    const resultIndicatorBox = document.querySelector('.result-indicator-box');
    const modelUsedVal = document.getElementById('model-used-val');
    
    const btnReset = document.getElementById('btn-reset');
    const btnCancelSelect = document.getElementById('btn-cancel-select');
    const modelButtons = document.querySelectorAll('.model-btn');

    // --- DEFENSA Y DIAGNÓSTICO DE INICIALIZACIÓN ---
    const requiredElements = {
        uploadView, modelSelectView, processingView, resultView,
        dropzone, fileInput, imagePreview, resultImage,
        progressFill, pipelineStatus, resultTitle, resultDesc,
        resultIndicatorBox, modelUsedVal, btnReset, btnCancelSelect
    };

    let missing = [];
    for (const [name, element] of Object.entries(requiredElements)) {
        if (!element) missing.push(name);
    }

    if (missing.length > 0) {
        console.error("Faltan elementos en el DOM:", missing);
        alert("Error de inicialización: Faltan elementos en la página HTML (" + missing.join(", ") + "). Por favor, limpia la caché de tu navegador y recarga la página (Ctrl + F5).");
        return;
    }

    // Estado local
    let uploadedImageSrc = null;
    let uploadedFile = null;
    let selectedModel = 'resnet50'; // ResNet-50 por defecto
    let apiResult = null;
    let apiError = false;

    // Nombres legibles para los modelos
    const modelNames = {
        'vit': 'Vision Transformer (ViT)',
        'efficientnetv2s': 'EfficientNetV2-S',
        'resnet50': 'ResNet-50',
        'mobilenetv2': 'MobileNetV2'
    };

    /* ----------------------------------------------------
       MANEJADORES DE EVENTOS DE SUBIDA DE IMAGEN
       ---------------------------------------------------- */

    // Prevenir comportamiento por defecto para Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Efectos visuales de arrastre
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    // Manejar la suelta del archivo
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleImageFile(files[0]);
        }
    });

    // Manejar la selección por clic
    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleImageFile(files[0]);
        }
    });

    // Procesar el archivo recibido y mostrar vista de selección de modelo
    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Por favor, selecciona un archivo de imagen válido (JPG, PNG).');
            return;
        }

        uploadedFile = file; // Guardar archivo crudo
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            uploadedImageSrc = reader.result;
            // Transicionar a la pantalla de selección de modelo
            switchView(uploadView, modelSelectView);
        };
    }

    /* ----------------------------------------------------
       SELECCIÓN DE MODELO
       ---------------------------------------------------- */
    modelButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            selectedModel = btn.getAttribute('data-model');
            console.log('Modelo seleccionado:', selectedModel);
            
            // Cargar previsualizaciones
            imagePreview.src = uploadedImageSrc;
            resultImage.src = uploadedImageSrc;

            // Cambiar de selección a procesamiento
            switchView(modelSelectView, processingView);
            startPipeline();
        });
    });

    btnCancelSelect.addEventListener('click', () => {
        // Limpiar inputs y volver
        fileInput.value = '';
        uploadedFile = null;
        uploadedImageSrc = null;
        switchView(modelSelectView, uploadView);
    });

    /* ----------------------------------------------------
       CONEXIÓN CON EL SERVIDOR DE INFERENCIA (FASTAPI)
       ---------------------------------------------------- */
    function fetchPrediction(file, model) {
        const formData = new FormData();
        formData.append('file', file);

        let primaryUrl = `/predict?modelo=${model}`;
        let fallbackUrl = `http://localhost:8000/predict?modelo=${model}`;

        // Ajustar URLs según el entorno para evitar retrasos por timeout
        if (window.location.protocol === 'file:') {
            primaryUrl = `http://localhost:8000/predict?modelo=${model}`;
            fallbackUrl = `http://localhost:8001/predict?modelo=${model}`;
        } else if (window.location.port !== '8000' && window.location.port !== '') {
            // Si el front corre en otro puerto (ej. Live Server en 5500), apuntar a FastAPI en 8000
            primaryUrl = `http://localhost:8000/predict?modelo=${model}`;
            fallbackUrl = `/predict?modelo=${model}`;
        }

        console.log(`Enviando consulta de predicción a: ${primaryUrl}`);

        return fetch(primaryUrl, { method: 'POST', body: formData })
            .catch(err => {
                console.log(`Puerto primario inactivo o error de red, reintentando en: ${fallbackUrl}`);
                return fetch(fallbackUrl, { method: 'POST', body: formData });
            })
            .then(res => {
                if (!res.ok) throw new Error('Respuesta no válida del pipeline');
                return res.json();
            });
    }

    /* ----------------------------------------------------
       EJECUCIÓN DEL PIPELINE DE PREPROCESAMIENTO
       ---------------------------------------------------- */
    function startPipeline() {
        // Resetear estados de la API
        apiResult = null;
        apiError = false;

        // Resetear barra de progreso
        progressFill.style.width = '0%';
        pipelineStatus.textContent = 'Iniciando preprocesamiento...';

        // Disparar la petición de inferencia en paralelo
        fetchPrediction(uploadedFile, selectedModel)
            .then(data => {
                console.log('Resultado del pipeline recibido:', data);
                apiResult = data;
            })
            .catch(err => {
                console.warn('Inferencia fallida, activando fallback local simulado:', err);
                apiError = true;
            });

        // Secuencias visuales de preprocesamiento
        const modelLabel = modelNames[selectedModel] || selectedModel;
        const steps = [
            { limit: 35, text: 'Paso 1/3: Detección y segmentación facial (YuNet/Haar, máscara elíptica en óvalo)...' },
            { limit: 70, text: 'Paso 2/3: Ajuste y mejora de contraste (Redimensionamiento y CLAHE en canal Y)...' },
            { limit: 90, text: `Paso 3/3: Normalización e inferencia en tensor PyTorch usando ${modelLabel}...` },
            { limit: 100, text: 'Estimando probabilidad final de presencia de acné...' }
        ];

        let currentProgress = 0;
        let stepIndex = 0;

        const interval = setInterval(() => {
            // Avanzar progreso visual
            currentProgress += Math.floor(Math.random() * 5) + 3;
            
            if (currentProgress >= 100) {
                currentProgress = 100;
                progressFill.style.width = '100%';
                pipelineStatus.textContent = 'Alineando datos de predicción...';
                
                clearInterval(interval);
                checkCompletion();
            } else {
                progressFill.style.width = `${currentProgress}%`;
                
                if (stepIndex < steps.length - 1 && currentProgress >= steps[stepIndex].limit) {
                    stepIndex++;
                }
                pipelineStatus.textContent = steps[stepIndex].text;
            }
        }, 100);

        function checkCompletion() {
            if (apiResult !== null) {
                setTimeout(() => {
                    generateDiagnosis(apiResult);
                }, 400);
            } else if (apiError) {
                setTimeout(() => {
                    generateDiagnosis(null); // Fallback simulado
                }, 400);
            } else {
                pipelineStatus.textContent = 'Finalizando análisis en el servidor de inferencia...';
                setTimeout(checkCompletion, 100); // Volver a chequear en 100ms
            }
        }
    }

    /* ----------------------------------------------------
       GENERACIÓN DE DIAGNÓSTICO REAL O SIMULADO
       ---------------------------------------------------- */
    function generateDiagnosis(result) {
        // Limpiar clases anteriores
        resultIndicatorBox.classList.remove('detected', 'free');

        const modelLabel = modelNames[selectedModel] || selectedModel;

        // Validar si el backend retornó un error o datos de inferencia vacíos/inválidos
        if (result !== null && (result.error || result.prob_acne === undefined || result.prob_acne === null || isNaN(result.prob_acne))) {
            resultIndicatorBox.classList.add('free'); // Usar clase neutra
            resultTitle.textContent = "Error de Análisis";
            
            let errMsg = (result && result.error) ? result.error : "No se pudo obtener una predicción numérica válida del modelo.";
            if (errMsg.toLowerCase().includes("imagen no válida")) {
                errMsg = "La imagen no es válida o no se pudo decodificar. Por favor, asegúrate de subir una foto en formato JPG o PNG (las imágenes HEIC de iPhone no son compatibles directamente).";
            }
            resultDesc.textContent = errMsg;
            modelUsedVal.textContent = modelLabel;
            
            // Transicionar a la pantalla de resultados para mostrar el error
            switchView(processingView, resultView);
            return;
        }

        let hasAcne = false;
        let isSimulated = false;
        let faceDetected = true;
        let prob = 0.5;

        if (result !== null) {
            // Usar datos reales del modelo de PyTorch
            hasAcne = result.label === 'acne';
            faceDetected = result.face_detected !== false;
            prob = result.prob_acne;
            modelUsedVal.textContent = modelLabel;
        } else {
            // Modo simulación si el backend no está disponible
            hasAcne = Math.random() < 0.5;
            isSimulated = true;
            prob = hasAcne ? (0.5 + Math.random() * 0.45) : (Math.random() * 0.45);
            modelUsedVal.textContent = `${modelLabel} (Simulación Local)`;
        }

        // Calcular el porcentaje de certeza/probabilidad de la predicción realizada
        const percentage = Math.round((hasAcne ? prob : (1.0 - prob)) * 100);

        if (hasAcne) {
            resultIndicatorBox.classList.add('detected');
            resultTitle.textContent = `Acné Detectado (${percentage}%)`;
            
            let desc = `El análisis procesado por el pipeline del modelo ${modelLabel} indica la presencia de lesiones activas de acné en el rostro con un ${percentage}% de probabilidad.`;
            if (isSimulated) {
                desc += ' (Nota: El servidor de inferencia local está inactivo).';
            }
            resultDesc.textContent = desc;
        } else {
            resultIndicatorBox.classList.add('free');
            resultTitle.textContent = `Sin Acné Detectado (${percentage}%)`;
            
            let desc = `El análisis procesado por el pipeline del modelo ${modelLabel} no observa signos o brotes inflamatorios significativos de acné en tu piel con un ${percentage}% de probabilidad.`;
            if (isSimulated) {
                desc += ' (Nota: El servidor de inferencia local está inactivo).';
            }
            resultDesc.textContent = desc;
        }

        // Cambiar a vista de resultados
        switchView(processingView, resultView);
    }

    /* ----------------------------------------------------
       REINICIO
       ---------------------------------------------------- */
    btnReset.addEventListener('click', () => {
        // Limpiar inputs
        fileInput.value = '';
        uploadedImageSrc = null;
        uploadedFile = null;
        imagePreview.src = '';
        resultImage.src = '';

        // Regresar a vista de carga
        switchView(resultView, uploadView);
    });

    /* ----------------------------------------------------
       UTILIDADES DE NAVEGACIÓN
       ---------------------------------------------------- */
    function switchView(fromView, toView) {
        if (!fromView || !toView) {
            console.error("switchView: Vistas no válidas", { fromView, toView });
            return;
        }
        fromView.classList.remove('active');
        setTimeout(() => {
            fromView.style.display = 'none';
            toView.style.display = 'block';
            setTimeout(() => {
                toView.classList.add('active');
            }, 50);
        }, 300);
    }
});
