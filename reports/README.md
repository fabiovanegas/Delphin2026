# Reports - Delphin 2026

Esta carpeta contiene los reportes, documentos y presentaciones generados durante el Programa Delfín 2026.

---

## 📄 Reportes por Semana

| Semana | Archivo | Descripción | Estado |
|--------|---------|-------------|--------|
| **Semana 1** | `1. Resultados simulacion Delphin_semana1.pdf` | Exploración inicial del corpus DAIC-WOZ, análisis de transcripciones, etiquetas PHQ-8 y visualización de señales de audio | ✅ Completado |
| **Semana 2** | `2. Procesamiento audio Semana 2.pdf` | Extracción de 186 características acústicas (MFCC, Delta, Delta-Delta, Chroma, Mel, ZCR, RMS). Entrenamiento de MLP para clasificación de audio (F1 Macro = 0.364) | ✅ Completado |
| **Semana 3** | `3. Semana 3 Tratamiento Texto .pdf` | Procesamiento de texto con BERT Base Uncased. Generación de embeddings de 768 dimensiones. Entrenamiento de MLP-BERT (F1 Macro = 0.190) | ✅ Completado |
| **Semana 4** | `4. Semana 4 Comparativa Baselines.pdf` | Comparativa de 9 modelos (SVM, RF, XGBoost, MLP) en audio y texto. Análisis de errores, curvas ROC, t-SNE, ablación y SHAP | ✅ Completado |
| **Semana 5** | `5. Semana 5 Fusión Multimodal (Audio + Texto).pdf` | Estrategias de fusión: Late Fusion, Early Fusion + PCA (74 componentes) y Cross-Attention. Mejor modelo global: Cross-Attention (F1 Macro = 0.572) | ✅ Completado |

---

## 🎯 Contenido de los Reportes por Semana

### Semana 1: Exploración Inicial

- Montaje de Google Drive y configuración del entorno
- Exploración del corpus DAIC-WOZ (1716 archivos)
- Carga y análisis de etiquetas PHQ-8 (35 participantes en DEV)
- Análisis de balance de clases (23 controles, 12 depresión)
- Distribución de puntajes PHQ-8 y género
- Carga y análisis de transcripciones (intervenciones del participante)
- Análisis de señales de audio (duración, forma de onda, espectrograma)

### Semana 2: Procesamiento de Audio

- **Metodología estricta**: TRAIN (107) y DEV (35) oficiales de DAIC-WOZ
- **Extracción de características** (186 features):
  - MFCC (13 coeficientes)
  - Delta y Delta-Delta
  - Chroma STFT (12 bandas)
  - Mel Spectrogram (40 bandas)
  - Zero Crossing Rate (ZCR)
  - RMS Energy
- **Agregación**: Media y desviación estándar por coeficiente/banda
- **Escalado**: StandardScaler ajustado solo en TRAIN
- **Modelo**: MLP (186 → 128 → 64 → 1) con Dropout (0.3)
- **Resultados en DEV**:
  - Accuracy: 0.600
  - F1 Macro: 0.364
  - Matriz de confusión: [[17, 6], [8, 4]]
- **Productos generados**: `audio_dataset.csv`, `X_train_audio.npy`, `X_dev_audio.npy`, `audio_scaler.pkl`, `audio_mlp_best.pth`

### Semana 3: Procesamiento de Texto con BERT

- **Metodología**: Mismos splits oficiales (TRAIN 107, DEV 35)
- **Modelo**: `bert-base-uncased` (congelado)
- **Estrategia**: Embedding del token [CLS] (768 dimensiones)
- **Limpieza de texto**:
  - Conversión a minúsculas
  - Normalización de espacios
  - Conservación de disfluencias (um, uh, hmm) por su valor clínico potencial
- **Límite de tokens**: 512 (truncamiento)
- **Modelo**: MLP (768 → 256 → 64 → 1) con Dropout (0.3)
- **Resultados en DEV**:
  - Accuracy: 0.514
  - F1 Macro: 0.190
  - Matriz de confusión: [[16, 7], [10, 2]]
- **Productos generados**: `texto_dataset.csv`, `X_train_text.npy`, `X_dev_text.npy`, `scaler_text.pkl`, `text_mlp_best.pth`

### Semana 4: Comparativa de Baselines

- **Modelos evaluados**: 9 en total
  - SVM Lineal y RBF (audio y texto)
  - Random Forest (audio y texto)
  - XGBoost (audio y texto)
  - MLP y MLP-BERT (de semanas anteriores)
- **Métricas**: F1 Macro, AUC-ROC, Sensibilidad, Especificidad
- **Análisis realizados**:
  - Curvas ROC (mejores modelos de audio y texto)
  - Análisis de errores (FN/FP) del mejor modelo
  - Concordancia entre modalidades (Cohen's Kappa = 0.094)
  - Mapa de errores por participante (identificación de "hard cases")
  - t-SNE de embeddings (audio y texto)
  - Ablación de features de audio
  - SHAP sobre el mejor modelo basado en árboles
- **Mejores modelos**:
  - Audio: SVM Lineal (F1 Macro = 0.450, Accuracy = 0.660)
  - Texto: XGBoost (F1 Macro = 0.240, Accuracy = 0.630)

### Semana 5: Fusión Multimodal

- **Estrategias implementadas**:
  - **Late Fusion**: Combinación de probabilidades (MLP+MLP, SVM+XGBoost)
  - **Early Fusion**: Concatenación de features (954 dims) + PCA (74 componentes)
  - **Cross-Attention**: Mecanismo de atención cruzada entre modalidades
- **Resultados principales**:
  - Cross-Attention: F1 Macro = **0.572** (mejor modelo global)
  - Early Fusion + PCA: F1 Macro = 0.470, AUC = 0.544
  - Late Fusion (MLP+MLP): F1 Macro = 0.364
  - Late Fusion (SVM+XGBoost): F1 Macro = 0.240
- **Análisis complementarios**:
  - Participantes "rescatados" y "perdidos" por la fusión
  - Ablación de modalidades (Solo Audio: 0.354, Solo Texto: 0.514, Ambos: 0.470)
  - SHAP sobre Early Fusion + PCA
- **Productos generados**: `early_fusion_metrics.json`, `cross_attention_metrics.json`, `tabla_final_resultados_proyecto_delphin.csv`

---

## 📊 Resumen de Resultados Finales

| Semana | Modalidad | Mejor Modelo | Accuracy | F1 Macro | AUC-ROC |
|--------|-----------|--------------|----------|----------|---------|
| 2 | Audio | MLP | 0.600 | 0.364 | 0.402 |
| 3 | Texto | MLP-BERT | 0.514 | 0.190 | 0.471 |
| 4 | Audio | SVM Lineal | 0.660 | 0.450 | 0.540 |
| 4 | Texto | XGBoost | 0.630 | 0.240 | 0.420 |
| 5 | Multimodal | Cross-Attention | — | **0.572** | — |
| 5 | Multimodal | Early Fusion + PCA | 0.543 | 0.470 | 0.544 |

**Mejor modelo global del proyecto**: Cross-Attention con F1 Macro = 0.572

---

## 👤 Autor

**Fabio Augusto Vanegas Bovea**  
Programa Delfín 2026  
IPN UPIIT Tlaxcala

---

*Última actualización: 22 de julio de 2026*



## 📁 Estructura de la Carpeta

