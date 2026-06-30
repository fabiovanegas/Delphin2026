# Delphin 2026 - Análisis Multimodal de Voz y Texto

## Descripción

Proyecto desarrollado en el Programa Delfín 2026 orientado al análisis multimodal de señales de voz y texto para la detección de indicadores de depresión y angustia psicológica utilizando el corpus DAIC-WOZ.

## Objetivos

* Analizar el corpus DAIC-WOZ.
* Extraer características acústicas y textuales.
* Implementar modelos de aprendizaje automático para la detección de depresión.
* Comparar resultados con benchmarks reportados en la literatura científica.
* Desarrollar un pipeline reproducible para investigación en salud mental computacional.

## Dataset

DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz)

Modalidades utilizadas:

* Audio (.wav)
* Transcripciones (.csv)

Nota: El dataset no se encuentra incluido en este repositorio debido a restricciones de tamaño y licenciamiento.

## Estructura del Proyecto

```text
Delphin2026/
│
├── notebooks/
├── src/
├── reports/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Tecnologías

* Python 3.10+
* Pandas
* NumPy
* Scikit-Learn
* PyTorch
* HuggingFace Transformers
* Librosa
* Google Colab
* Git y GitHub

## Instalación

```bash
pip install -r requirements.txt
```
## Avance

### Semana 1
- Configuración de Google Colab.
- Exploración del corpus DAIC-WOZ.
- Análisis de transcripciones.
- Análisis de audio.
- Exploración de etiquetas PHQ-8.
- Configuración de GitHub.

### Semana 2
- Revisión de benchmarks del área.
- Estudio de características acústicas.
- Exploración de COVAREP y FORMANT.
- Preparación para extracción de características.

# Semana 3 – Procesamiento de Texto con BERT

## Descripción

En esta etapa del proyecto se desarrolló el procesamiento de la modalidad textual del conjunto de datos DAIC-WOZ utilizando **BERT Base Uncased** para obtener representaciones vectoriales (embeddings) de las entrevistas. Posteriormente, estos embeddings fueron utilizados para entrenar y evaluar un clasificador de Regresión Logística orientado a la detección de depresión.

---

## Objetivos

- Procesar los transcript de las entrevistas.
- Generar embeddings de 768 dimensiones utilizando BERT Base.
- Construir un clasificador basado en Regresión Logística.
- Evaluar el desempeño mediante métricas de clasificación.
- Analizar el efecto del límite de 512 tokens de BERT.
- Comparar diferentes estrategias de extracción de embeddings.

---

## Metodología

### 1. Carga de los transcript

Se utilizaron los transcript del conjunto de datos DAIC-WOZ correspondientes a las entrevistas realizadas entre Ellie y los participantes.

---

### 2. Preprocesamiento del texto

Se realizó una limpieza básica del texto que incluyó:

- Conversión a minúsculas.
- Eliminación de espacios innecesarios.
- Conservación del contenido lingüístico relevante.

Las disfluencias (por ejemplo, *um*, *uh* y *hmm*) se conservaron en el texto debido a su posible relación con indicadores de depresión reportados en la literatura.

---

### 3. Generación de embeddings

Se utilizó el modelo:

- **bert-base-uncased**

Los transcript fueron tokenizados utilizando:

- `truncation=True`
- `max_length=512`

Cada entrevista fue representada mediante un vector de **768 características**.

---

### 4. Análisis del límite de 512 tokens

Se calculó el número de tokens de cada transcript antes del proceso de truncamiento.

Los resultados mostraron que una proporción importante de las entrevistas supera el límite de 512 tokens permitido por BERT Base.

Por esta razón se utilizó truncamiento, conservando únicamente los primeros 512 tokens de cada entrevista.

Como trabajo futuro se propone evaluar estrategias como:

- división del transcript en bloques (chunks),
- modelos para secuencias largas (Longformer).

---

### 5. Construcción del conjunto de datos

Los embeddings generados fueron integrados con las etiquetas clínicas del conjunto DAIC-WOZ utilizando el identificador `Participant_ID`.

Se construyeron los conjuntos:

- Train (Development)
- Test

manteniendo la misma partición utilizada durante el procesamiento de audio.

---

### 6. Clasificación

Se entrenó un modelo de:

- Regresión Logística

utilizando como variables predictoras los embeddings de BERT.

---

### 7. Evaluación

Se calcularon las siguientes métricas:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

Además se generaron:

- Matriz de confusión.
- Curva ROC.

---

## Comparación de estrategias de embeddings

Se compararon dos estrategias para representar los transcript:

1. Token **CLS**.
2. **Mean Pooling** de la última capa de BERT.

Los resultados obtenidos fueron:

| Representación | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------|---------:|----------:|--------:|---------:|--------:|
| CLS | 0.6383 | 0.3636 | 0.2857 | 0.3200 | 0.5996 |
| Mean Pooling | 0.6170 | 0.3333 | 0.2857 | 0.3077 | 0.6255 |

La representación basada en **CLS** obtuvo el mejor equilibrio entre Accuracy, Precision y F1-Score, mientras que **Mean Pooling** presentó un AUC-ROC ligeramente superior.

Para las siguientes etapas del proyecto se seleccionó la representación basada en **CLS**.

---

## Limitaciones

Se evaluó la posibilidad de utilizar **MentalBERT**, un modelo especializado en lenguaje relacionado con salud mental.

Sin embargo, no fue posible incorporarlo al entorno de desarrollo debido a problemas de disponibilidad y compatibilidad durante la ejecución en Google Colab.

Por esta razón se utilizó **BERT Base Uncased**, garantizando la reproducibilidad del experimento.

---

## Archivos generados

- `Embeddings_BERT.csv`
- Embeddings utilizando CLS.
- Embeddings utilizando Mean Pooling.
- Resultados del clasificador.
- Matriz de confusión.
- Curva ROC.
- Comparación de estrategias de embeddings.

---

## Tecnologías utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Matplotlib

---

## Próxima etapa

En la **Semana 4** se desarrollará el modelo multimodal mediante la integración de las modalidades de audio y texto, comparando el desempeño de:

- Solo audio.
- Solo texto.
- Audio + texto.




## Autor

Fabio Augusto Vanegas Bovea

Programa Delfín 2026
IPN UPIIT Tlaxcala
