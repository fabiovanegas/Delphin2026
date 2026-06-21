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

## Autor

Fabio Augusto Vanegas Bovea

Programa Delfín 2026
IPN UPIIT Tlaxcala
