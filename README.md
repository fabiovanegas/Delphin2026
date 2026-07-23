Delphin 2026 - Análisis Multimodal de Voz y Texto
Descripción
Proyecto desarrollado en el Programa Delfín 2026 orientado al análisis multimodal de señales de voz y texto para la detección de indicadores de depresión y angustia psicológica utilizando el corpus DAIC-WOZ.

Objetivos
Analizar el corpus DAIC-WOZ.

Extraer características acústicas y textuales.

Implementar modelos de aprendizaje automático para la detección de depresión.

Comparar resultados con benchmarks reportados en la literatura científica.

Desarrollar un pipeline reproducible para investigación en salud mental computacional.

Dataset
DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)

Modalidades utilizadas:

Audio (.wav)

Transcripciones (.csv)

Nota: El dataset no se encuentra incluido en este repositorio debido a restricciones de tamaño y licenciamiento.

Estructura del Proyecto
Delphin2026/
├── notebooks/
│ ├── Delphin_semana1.ipynb
│ ├── Procesamiento_audio_Semana_2.ipynb
│ ├── 3_Semana_3_Tratamiento_Texto.ipynb
│ ├── 4_Semana_4_Comparativa_Baselines.ipynb
│ └── 5_Semana_5_Fusion_Multimodal.ipynb
├── src/
├── reports/
├── results/
├── README.md
├── requirements.txt
└── .gitignore

Tecnologías
Python 3.10+

Pandas

NumPy

Scikit-Learn

PyTorch

HuggingFace Transformers

XGBoost

Librosa

Google Colab

Git y GitHub

Instalación
pip install -r requirements.txt

Avance del Proyecto
Semana 1: Exploración Inicial
Configuración de Google Colab.

Exploración del corpus DAIC-WOZ.

Análisis de transcripciones.

Análisis de audio.

Exploración de etiquetas PHQ-8.

Configuración de GitHub.

Semana 2: Procesamiento de Audio
Revisión de benchmarks del área.

Estudio de características acústicas.

Extracción de características con COVAREP y FORMANT.

Entrenamiento de MLP para clasificación de audio.

Generación de matrices: X_train_audio.npy, X_dev_audio.npy (186 features).

Semana 3: Procesamiento de Texto con BERT
Generación de embeddings con BERT Base Uncased (768 dimensiones).

Evaluación de estrategias: CLS vs Mean Pooling.

Entrenamiento de MLP para clasificación de texto.

Generación de matrices: X_train_text.npy, X_dev_text.npy.

Resultados de estrategias de embeddings:

Representación	Accuracy	Precision	Recall	F1-Score	AUC-ROC
CLS	0.6383	0.3636	0.2857	0.3200	0.5996
Mean Pooling	0.6170	0.3333	0.2857	0.3077	0.6255
La representación basada en CLS fue seleccionada para las siguientes etapas.

Limitación: No fue posible utilizar MentalBERT debido a problemas de compatibilidad en Google Colab.

Semana 4: Comparativa de Baselines
Evaluación de 9 modelos: SVM (lineal/RBF), Random Forest, XGBoost, MLP.

Comparativa entre clasificadores clásicos y redes neuronales.

Análisis de errores por participante (FN/FP).

Curvas ROC y visualización t-SNE.

Ablación de features de audio (MFCC, Delta, Chroma, Mel, ZCR, RMS).

SHAP para interpretabilidad del mejor modelo basado en árboles.

Análisis de concordancia entre modalidades (Cohen's Kappa).

Resultados principales:

Modelo	Modalidad	Accuracy	F1 Macro	AUC-ROC
SVM Lineal	Audio	0.660	0.450	0.540
XGBoost	Texto	0.630	0.240	0.420
MLP	Audio	0.600	0.364	0.402
MLP-BERT	Texto	0.514	0.190	0.471
Mejor modelo de la semana: SVM Lineal (Audio) con F1 Macro = 0.450

Semana 5: Fusión Multimodal
Late Fusion: Combinación de probabilidades (MLP+MLP, SVM+XGBoost).

Early Fusion: Concatenación de features (954 dims) + PCA (74 componentes).

Cross-Attention: Mecanismo de atención cruzada entre modalidades.

Análisis de participantes "rescatados" y "perdidos" por la fusión.

Ablación de modalidades (aporte de audio vs texto).

Resultados principales:

Estrategia	Modelo	Accuracy	F1 Macro	AUC-ROC
Cross-Attention	CrossAttn	—	0.572	—
Early Fusion	XGBoost+PCA	0.543	0.470	0.544
Late Fusion	MLP+MLP	0.600	0.364	0.402
Late Fusion	SVM+XGBoost	0.630	0.240	0.420
Mejor modelo global del proyecto: Cross-Attention con F1 Macro = 0.572

Resumen de Resultados Finales
Semana	Modalidad	Mejor Modelo	F1 Macro
2	Audio	MLP	0.364
3	Texto	MLP-BERT	0.190
4	Audio	SVM Lineal	0.450
4	Texto	XGBoost	0.240
5	Multimodal	Cross-Attention	0.572
Conclusiones
El modelo Cross-Attention logró el mejor rendimiento global (F1 Macro = 0.572), superando a los enfoques unimodales.

La fusión multimodal demostró ser efectiva, especialmente en la recuperación de participantes que los modelos individuales no clasificaban correctamente.

El análisis de concordancia (Cohen's Kappa = 0.094) entre audio y texto sugiere que los errores son complementarios, lo que favorece la fusión multimodal.

La ablación de features mostró que los componentes de audio como MFCC y Delta son los más informativos.

El análisis SHAP reveló que ciertas dimensiones del embedding BERT son más influyentes para la clasificación.

Trabajo Futuro
Evaluar estrategias para secuencias largas (Longformer) en procesamiento de texto.

Incorporar MentalBERT una vez que sea compatible con el entorno.

Realizar pruebas en el conjunto TEST (47 participantes) para validación final.

Explorar arquitecturas más complejas de fusión multimodal.

Desarrollar una interfaz para visualización de resultados clínicos.

Autor
Fabio Augusto Vanegas Bovea

Programa Delfín 2026
IPN UPIIT Tlaxcala

Referencias
Gratch, J., et al. (2014). "The Distress Analysis Interview Corpus of human and computer interviews"

DAIC-WOZ Dataset: https://dcapswoz.ict.usc.edu/

Programa Delfín 2026: https://www.programadelfin.org.mx/
