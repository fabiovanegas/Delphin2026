````markdown
# DELPHIN 2026 — Análisis Multimodal de Voz y Texto

## Descripción

Este proyecto fue desarrollado en el marco del **Programa Delfín 2026** y está orientado al análisis multimodal de señales de **audio y texto** para la detección computacional de indicadores asociados con **depresión y angustia psicológica**.

El estudio utiliza el corpus **DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz)** y explora diferentes enfoques de aprendizaje automático y aprendizaje profundo aplicados de manera unimodal y multimodal.

El proyecto comprende el procesamiento de audio, el análisis de transcripciones mediante representaciones BERT, la evaluación de modelos de clasificación y la implementación de diferentes estrategias de fusión multimodal.

---

## Objetivos

Los principales objetivos del proyecto son:

- Analizar y comprender la estructura del corpus DAIC-WOZ.
- Procesar las modalidades de audio y texto disponibles en el dataset.
- Extraer características acústicas relevantes de las señales de voz.
- Generar representaciones vectoriales de texto mediante modelos basados en BERT.
- Implementar y comparar modelos de aprendizaje automático y redes neuronales para la clasificación de depresión.
- Evaluar estrategias de fusión multimodal que integren información de audio y texto.
- Comparar los resultados obtenidos con benchmarks reportados en la literatura científica.
- Desarrollar un pipeline reproducible orientado a la investigación en salud mental computacional.

---

## Dataset

El proyecto utiliza el corpus:

**DAIC-WOZ — Distress Analysis Interview Corpus: Wizard of Oz**

Este dataset contiene entrevistas clínicas multimodales diseñadas para apoyar investigaciones relacionadas con la detección automática de condiciones de salud mental.

### Modalidades utilizadas

- **Audio:** archivos `.wav`.
- **Texto:** transcripciones de las entrevistas.
- **Etiquetas clínicas:** información asociada con PHQ-8 utilizada para definir las clases de clasificación.

> **Nota:** El dataset DAIC-WOZ no se encuentra incluido en este repositorio debido a restricciones de licenciamiento, privacidad y tamaño. Para utilizarlo es necesario solicitar acceso a través de los canales oficiales correspondientes.

---

## Estructura del Proyecto

```text
Delphin2026/
│
├── notebooks/
│   ├── Delphin_semana1.ipynb
│   ├── Procesamiento_audio_Semana_2.ipynb
│   ├── 3_Semana_3_Tratamiento_Texto.ipynb
│   ├── 4_Semana_4_Comparativa_Baselines.ipynb
│   └── 5_Semana_5_Fusion_Multimodal.ipynb
│
├── src/
├── reports/
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
````

### Descripción de las carpetas

* `notebooks/`: contiene los notebooks correspondientes a cada etapa del proyecto.
* `src/`: destinado a funciones, módulos y código reutilizable.
* `reports/`: documentación, informes y material asociado con los resultados del proyecto.
* `results/`: resultados experimentales, métricas, gráficas y archivos generados durante los experimentos.
* `requirements.txt`: dependencias necesarias para reproducir el entorno.
* `.gitignore`: configuración de archivos y directorios excluidos del control de versiones.

---

## Tecnologías Utilizadas

El proyecto fue desarrollado principalmente con:

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* XGBoost
* Librosa
* Google Colab
* Git
* GitHub

---

## Instalación

Para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Se recomienda utilizar un entorno virtual o ejecutar los notebooks mediante **Google Colab** para facilitar la reproducibilidad del entorno experimental.

---

# Desarrollo del Proyecto

## Semana 1 — Exploración Inicial

Durante la primera etapa se realizó la exploración y comprensión inicial del corpus DAIC-WOZ.

### Actividades realizadas

* Configuración del entorno de trabajo en Google Colab.
* Exploración de la estructura del corpus DAIC-WOZ.
* Análisis inicial de las transcripciones.
* Exploración de los archivos de audio.
* Revisión de las etiquetas clínicas PHQ-8.
* Organización inicial del repositorio.
* Configuración del control de versiones mediante Git y GitHub.

Esta etapa permitió establecer la estructura general del proyecto y preparar los datos para las fases posteriores de procesamiento.

---

## Semana 2 — Procesamiento de Audio

La segunda etapa se centró en el procesamiento de la modalidad de audio y en la extracción de características acústicas relevantes para la clasificación.

### Actividades realizadas

* Revisión de benchmarks y trabajos relacionados.
* Exploración de características acústicas utilizadas en detección automática de depresión.
* Procesamiento de señales de audio.
* Extracción de características acústicas.
* Preparación de matrices de características para entrenamiento y validación.
* Entrenamiento de modelos para clasificación basada en audio.

Entre las características acústicas estudiadas se incluyeron:

* **MFCC (Mel-Frequency Cepstral Coefficients).**
* **Delta MFCC.**
* **Delta-Delta MFCC.**
* **Chroma.**
* **ZCR (Zero Crossing Rate).**
* **RMS Energy.**

Se generaron matrices de características como:

```text
X_train_audio.npy
X_dev_audio.npy
```

Cada participante quedó representado mediante un vector de aproximadamente **186 características acústicas**.

Los experimentos de esta etapa sirvieron como base para los posteriores modelos unimodales de audio y para las estrategias de fusión multimodal.

---

## Semana 3 — Procesamiento de Texto con BERT

La tercera etapa se enfocó en el procesamiento de las transcripciones mediante modelos Transformer.

Se utilizó:

**BERT Base Uncased**

para generar representaciones vectoriales de las entrevistas.

Cada representación textual fue codificada mediante vectores de:

**768 dimensiones**

### Estrategias evaluadas

Se analizaron principalmente dos estrategias de representación:

* **CLS Token Representation**
* **Mean Pooling**

### Resultados de las estrategias de representación

| Representación | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| -------------- | -------: | --------: | -----: | -------: | ------: |
| CLS            |   0.6383 |    0.3636 | 0.2857 |   0.3200 |  0.5996 |
| Mean Pooling   |   0.6170 |    0.3333 | 0.2857 |   0.3077 |  0.6255 |

La representación basada en **CLS** fue seleccionada como representación principal para continuar con las siguientes etapas experimentales.

También se generaron matrices como:

```text
X_train_text.npy
X_dev_text.npy
```

Posteriormente, estas representaciones fueron utilizadas como entrada para diferentes modelos de clasificación.

### Limitación experimental

Se consideró el uso de modelos especializados en lenguaje relacionado con salud mental, como **MentalBERT**. Sin embargo, su integración no pudo completarse satisfactoriamente dentro del entorno experimental utilizado.

Por esta razón, se mantuvo **BERT Base Uncased** como modelo principal para la generación de embeddings textuales.

---

## Semana 4 — Comparación de Modelos Baseline

Durante esta etapa se realizó una comparación sistemática de diferentes algoritmos de aprendizaje automático aplicados a las modalidades de audio y texto.

### Modelos evaluados

Entre los modelos analizados se encuentran:

* **Support Vector Machine — SVM Lineal**
* **Support Vector Machine — SVM RBF**
* **Random Forest**
* **XGBoost**
* **Multi-Layer Perceptron — MLP**

Se evaluaron diferentes configuraciones hasta completar una comparación amplia de modelos unimodales.

### Análisis realizados

Además de las métricas tradicionales de clasificación, se realizaron:

* Comparación entre clasificadores clásicos y redes neuronales.
* Análisis de errores por participante.
* Identificación de falsos positivos y falsos negativos.
* Análisis mediante curvas ROC.
* Visualización de representaciones mediante t-SNE.
* Ablación de características de audio.
* Interpretabilidad mediante SHAP.
* Análisis de concordancia entre modalidades mediante Cohen's Kappa.

### Resultados representativos

| Modelo     | Modalidad | Accuracy |    F1 | AUC-ROC |
| ---------- | --------- | -------: | ----: | ------: |
| SVM Lineal | Audio     |    0.660 | 0.450 |   0.540 |
| XGBoost    | Texto     |    0.630 | 0.240 |   0.420 |
| MLP        | Audio     |    0.600 | 0.364 |   0.402 |
| MLP-BERT   | Texto     |    0.514 | 0.190 |   0.471 |

Entre los modelos unimodales evaluados, **SVM Lineal aplicado a características de audio** presentó uno de los mejores desempeños globales de esta etapa.

Estos resultados también mostraron diferencias importantes entre el comportamiento de las modalidades de audio y texto, motivando la exploración posterior de estrategias de fusión multimodal.

---

## Semana 5 — Fusión Multimodal

La quinta etapa exploró diferentes estrategias para combinar la información proveniente de las modalidades de **audio y texto**.

Se estudiaron tres enfoques principales:

### 1. Late Fusion

La fusión tardía combina las predicciones o probabilidades generadas independientemente por modelos de audio y texto.

Se evaluaron combinaciones como:

```text
MLP Audio + MLP Texto
SVM Audio + XGBoost Texto
```

Este enfoque permite mantener modelos independientes para cada modalidad y combinar posteriormente sus decisiones.

---

### 2. Early Fusion

En este enfoque se concatenaron directamente las representaciones de ambas modalidades.

Dimensiones originales:

```text
Audio: 186 características
Texto: 768 características

Total concatenado: 954 dimensiones
```

Posteriormente se aplicó reducción de dimensionalidad mediante **Principal Component Analysis (PCA)**.

La representación fue reducida aproximadamente a:

```text
74 componentes principales
```

Esto permitió disminuir la dimensionalidad antes de entrenar los clasificadores multimodales.

---

### 3. Cross-Attention

Finalmente, se exploró una arquitectura basada en **Cross-Attention**, diseñada para aprender interacciones entre las representaciones de audio y texto.

A diferencia de la simple concatenación, este mecanismo permite modelar interacciones entre ambas modalidades mediante un mecanismo de atención cruzada.

Este enfoque obtuvo el mejor resultado experimental global del proyecto en términos de F1 reportado.

### Resultados principales

| Estrategia      | Modelo        | Accuracy |        F1 | AUC-ROC |
| --------------- | ------------- | -------: | --------: | ------: |
| Cross-Attention | CrossAttn     |        — | **0.572** |       — |
| Early Fusion    | XGBoost + PCA |    0.543 |     0.470 |   0.544 |
| Late Fusion     | MLP + MLP     |    0.600 |     0.364 |   0.402 |
| Late Fusion     | SVM + XGBoost |    0.630 |     0.240 |   0.420 |

El enfoque **Cross-Attention** alcanzó el mejor desempeño experimental global reportado:

**F1 = 0.572**

---

# Resumen de Resultados Finales

La evolución experimental del proyecto puede resumirse de la siguiente manera:

| Semana | Modalidad  | Mejor Modelo    |        F1 |
| ------ | ---------- | --------------- | --------: |
| 2      | Audio      | MLP             |     0.364 |
| 3      | Texto      | MLP-BERT        |     0.190 |
| 4      | Audio      | SVM Lineal      |     0.450 |
| 4      | Texto      | XGBoost         |     0.240 |
| 5      | Multimodal | Cross-Attention | **0.572** |

Los resultados muestran una mejora del mejor rendimiento reportado al pasar de los enfoques unimodales evaluados hacia la estrategia multimodal basada en **Cross-Attention**.

---

# Análisis Multimodal

Uno de los principales hallazgos del proyecto fue que las modalidades de audio y texto presentan comportamientos diferentes durante la clasificación.

El análisis de concordancia mediante **Cohen's Kappa** mostró un valor aproximado de:

```text
κ = 0.094
```

Este bajo nivel de concordancia sugiere que los modelos de audio y texto tienden a cometer errores diferentes.

Desde una perspectiva multimodal, esta característica resulta relevante porque indica que ambas fuentes de información pueden contener señales complementarias.

La combinación de modalidades permitió analizar:

* Participantes correctamente clasificados por ambas modalidades.
* Participantes clasificados correctamente únicamente por audio.
* Participantes clasificados correctamente únicamente por texto.
* Participantes recuperados mediante estrategias de fusión.
* Participantes cuya clasificación empeoró después de la fusión.

Este análisis permitió estudiar no solamente las métricas globales, sino también el comportamiento de los modelos a nivel individual.

---

# Interpretabilidad

Se incorporaron técnicas de interpretabilidad para analizar qué variables influyen en las decisiones de los modelos.

Entre las técnicas utilizadas se encuentra:

**SHAP — SHapley Additive exPlanations**

Este análisis permitió explorar la contribución de diferentes características acústicas y dimensiones de las representaciones textuales.

Los experimentos sugieren que determinadas características acústicas, especialmente las relacionadas con **MFCC y sus variaciones temporales**, contienen información relevante para la clasificación.

En la modalidad textual, determinadas dimensiones de los embeddings generados mediante BERT mostraron una mayor influencia en las predicciones de algunos modelos.

---

# Conclusiones

Los experimentos realizados muestran que la combinación de información proveniente de **audio y texto** puede mejorar el desempeño frente a determinados modelos unimodales.

Entre los principales hallazgos se encuentran:

* El enfoque **Cross-Attention** obtuvo el mejor resultado experimental global reportado, con un **F1 de 0.572**.
* El modelo **SVM Lineal aplicado a audio** presentó un desempeño competitivo entre los modelos unimodales.
* Las modalidades de audio y texto mostraron patrones de error diferentes.
* El bajo nivel de concordancia entre modalidades sugiere que ambas contienen información potencialmente complementaria.
* Las estrategias de fusión multimodal permitieron aprovechar parcialmente esta complementariedad.
* Las características acústicas relacionadas con MFCC y sus variaciones temporales mostraron relevancia dentro de los análisis realizados.
* Las técnicas de interpretabilidad permitieron explorar la contribución de características acústicas y representaciones textuales en las decisiones de los modelos.

Los resultados obtenidos deben interpretarse considerando las limitaciones asociadas con el tamaño del conjunto de datos y la distribución de las clases.

Este proyecto constituye una exploración experimental del potencial de los sistemas multimodales para apoyar investigaciones en el área de **salud mental computacional**.

> **Importante:** Los modelos desarrollados tienen fines exclusivamente académicos y de investigación. No constituyen herramientas de diagnóstico clínico ni sustituyen la evaluación realizada por profesionales de la salud.

---

# Trabajo Futuro

Como continuación del proyecto se proponen las siguientes líneas de investigación:

* Evaluar modelos especializados en secuencias largas, como **Longformer**.
* Explorar modelos de lenguaje especializados en salud mental.
* Incorporar **MentalBERT** cuando sea posible integrarlo adecuadamente en el entorno experimental.
* Realizar pruebas sobre el conjunto **TEST** para una evaluación adicional del sistema.
* Implementar técnicas más avanzadas de fusión multimodal.
* Explorar arquitecturas Transformer multimodales.
* Analizar mecanismos de atención entre segmentos específicos de audio y texto.
* Evaluar estrategias adicionales para manejar el desbalance de clases.
* Aplicar esquemas de validación más robustos cuando el diseño experimental lo permita.
* Incorporar técnicas adicionales de explicabilidad e interpretabilidad.
* Desarrollar herramientas de visualización para facilitar el análisis de resultados experimentales.

---

# Reproducibilidad

Para facilitar la reproducción de los experimentos, el proyecto sigue de manera general el siguiente flujo:

```text
Datos
  ↓
Preprocesamiento
  ↓
Extracción de características
  ↓
Representaciones de Audio y Texto
  ↓
Modelos Unimodales
  ↓
Fusión Multimodal
  ↓
Evaluación
  ↓
Interpretabilidad
```

Los notebooks documentan progresivamente cada una de las etapas desarrolladas durante el proyecto.

Debido a las restricciones de acceso del corpus DAIC-WOZ, cada usuario debe obtener el dataset de manera independiente y configurar las rutas correspondientes antes de ejecutar los notebooks.

---

# Autor

**Fabio Augusto Vanegas Bovea**

**Programa Delfín 2026**
**IPN — UPIIT Tlaxcala**

Proyecto desarrollado con fines académicos y de investigación en el área de **Inteligencia Artificial, Aprendizaje Automático y Salud Mental Computacional**.

---

# Referencias

Gratch, J., Artstein, R., Lucas, G. M., Stratou, G., Scherer, S., Nazarian, A., Wood, R., Boberg, J., DeVault, D., Marsella, S., Traum, D., Rizzo, S., & Morency, L. P. (2014). *The Distress Analysis Interview Corpus of Human and Computer Interviews*. Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC).

**DAIC-WOZ — Distress Analysis Interview Corpus**

https://dcapswoz.ict.usc.edu/

**Programa Delfín**

https://www.programadelfin.org.mx/

---

# Aviso

Este repositorio corresponde a un proyecto académico de investigación.

Los resultados presentados son experimentales y dependen de las particiones, configuraciones, procedimientos de preprocesamiento y modelos utilizados durante el desarrollo del proyecto.

El sistema desarrollado **no está diseñado ni validado para realizar diagnósticos médicos o psicológicos** y no sustituye la evaluación de profesionales de la salud.

```
```
