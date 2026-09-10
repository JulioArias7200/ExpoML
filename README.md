# ML Ensemble Lab — Bagging vs Boosting

Aplicación Flask educativa que explica y compara **Bagging** (Random Forest) vs
**Boosting** (Gradient Boosting / AdaBoost) usando el **UCI Heart Disease Dataset**
(~920 filas de 4 centros médicos reales: Cleveland, Hungría, Suiza y Long Beach VA),
para clasificación de enfermedad cardíaca y regresión de variables clínicas.

## Requisitos

- Python 3.10+

## Instalación y ejecución

```bash
python -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Luego abre `http://localhost:5000`.

## Estructura

```
app.py                                    # rutas Flask + API JSON
ml/pipeline.py                            # carga de datos, entrenamiento, métricas y gráficas
templates/                                # páginas (Overview/Boosting, Bagging, Comparative, Demo)
static/plots/                             # gráficas generadas (se sobreescriben en cada entrenamiento)
data2/heart_disease/                      # dataset UCI Heart Disease (4 archivos processed.*.data)
  ├── processed.cleveland.data            # Cleveland Clinic (303 pacientes)
  ├── processed.hungarian.data            # Hungarian Institute of Cardiology
  ├── processed.switzerland.data          # University Hospital, Zurich
  └── processed.va.data                   # V.A. Medical Center, Long Beach
data/                                     # datasets adicionales, no usados por app.py actualmente
notebooks/                                # notebook exploratorio (no requerido para correr la app)
```

## Páginas y API

| Ruta                       | Qué hace                                                   |
|-----------------------------|-------------------------------------------------------------|
| `/`, `/boosting`            | Explicación teórica de Boosting                             |
| `/bagging`                  | Explicación teórica de Bagging                              |
| `/comparative`              | Tabla comparativa + panel de tuning **funcional** (entrena en vivo) |
| `/demo`                     | Demo interactiva: entrena y muestra todas las gráficas       |
| `GET /api/summary`          | Resumen del dataset (shape, nulos, balance de clases)        |
| `GET /api/train/classification` | Entrena RF / GB / AdaBoost, devuelve métricas + gráficas |
| `GET /api/train/regression`     | Entrena RF / GB para un target numérico                 |
| `GET /api/train/compare`        | Ejecuta clasificación + regresión juntas                 |

Parámetros aceptados por los endpoints de entrenamiento: `n_estimators`
(10–300), `learning_rate` (0.01–1.0, solo afecta a los modelos de boosting) y
`target` (solo en regresión: `Cholesterol Level`, `Max Heart Rate`,
`Resting Blood Pressure` o `Age`).

## Sobre el dataset UCI Heart Disease

El dataset utilizado es el **UCI Heart Disease Database**, un conjunto de datos
clínicos **reales** recolectados entre 1981-1984 de pacientes de cuatro centros
médicos internacionales:

- **Cleveland Clinic Foundation** (Ohio, EE.UU.) — 303 pacientes
- **Hungarian Institute of Cardiology** (Budapest, Hungría)
- **University Hospital** (Zurich, Suiza)  
- **V.A. Medical Center** (Long Beach, California)

**Fuente:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).  
Heart Disease [Dataset]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C52P4X

### Variables utilizadas (13 predictoras + 1 objetivo)

El pipeline usa las **14 variables estándar** del repositorio UCI:

**Variables numéricas (6):**
- Age (Edad en años)
- Resting Blood Pressure (Presión arterial en reposo, mm Hg)
- Cholesterol Level (Colesterol sérico, mg/dl)
- Max Heart Rate (Frecuencia cardíaca máxima alcanzada)
- ST Depression (Depresión del segmento ST inducida por ejercicio)
- Major Vessels Colored (Número de vasos principales coloreados, 0-3)

**Variables categóricas (7):**
- Sex (Sexo: Male/Female)
- Chest Pain Type (Tipo de dolor torácico: 4 categorías)
- Fasting Blood Sugar (Glucosa en ayunas > 120 mg/dl: Yes/No)
- Resting ECG (Resultados ECG en reposo: 3 categorías)
- Exercise Induced Angina (Angina inducida por ejercicio: Yes/No)
- ST Slope (Pendiente del segmento ST: 3 categorías)
- Thalassemia (Tipo de talasemia: 3 categorías)

**Variable objetivo:**
- **Heart Disease Status** (Yes/No) — derivada de la variable "num" original
  (0 = sin obstrucción significativa, 1-4 = distintos grados de severidad
  angiográfica; se convierte a binaria Yes/No para clasificación)

### Características del dataset

- **Total de filas:** ~920 (combinación de los 4 centros)
- **Desbalance de clases:** ~80% "No" / ~20% "Yes" (por eso el pipeline balancea
  con `class_weight="balanced"` para RF y `sample_weight` para GB/AdaBoost)
- **Valores faltantes:** Presentes, especialmente en variables que no todos los
  centros midieron. El preprocesamiento usa `SimpleImputer` con estrategias
  apropiadas (mediana para numéricas, moda para categóricas).
- **Cholesterol = 0:** Tratado como valor faltante (centinela usado por Suiza
  y Long Beach VA para "no medido").

### Resultados esperados

Al tratarse de datos clínicos reales con señal predictiva genuina, los modelos
alcanzan métricas superiores al azar:

- **Accuracy:** ~70-75% (superior al 50% de azar y al 80% de predecir siempre
  la clase mayoritaria sin aprendizaje real)
- **ROC AUC:** ~0.70-0.80 (muy superior al 0.5 de clasificador aleatorio)
- **R² (regresión):** Variable según el target; Max Heart Rate tiene correlación
  negativa fuerte con Age (~-0.4), permitiendo regresiones con R² positivos.

Estos resultados validan que:
1. El dataset contiene señal predictiva real de enfermedad cardíaca
2. Las técnicas de Bagging y Boosting capturan patrones clínicos significativos
3. La comparación entre ambos enfoques es metodológicamente válida

## Cambios recientes (mejoras de coherencia y funcionamiento)

- Se corrigió `plot_prob_distribution`: antes cortaba el ciclo tras el primer
  modelo (`return` dentro del `for`) y dejaba una figura de matplotlib sin
  cerrar; ahora genera un panel con las 3 distribuciones.
- Se balancearon las clases también para Boosting (`sample_weight`), ya que
  antes solo Random Forest usaba `class_weight="balanced"` y los modelos de
  boosting colapsaban a predecir siempre "No" (precision/recall = 0).
- Se acotó `learning_rate` (0.01–1.0) y `n_estimators` (10–300) en la API para
  evitar entrenamientos inválidos o excesivamente lentos.
- Se cacheó la lectura del CSV para acelerar entrenamientos repetidos desde
  el demo.
- El panel de "Tuning" de `/comparative` ahora entrena de verdad contra la
  API (antes los sliders y el botón no estaban conectados a nada).
- Se unificó el nombre del proyecto ("ML Ensemble Lab") en toda la interfaz.
- Se eliminó `templates/partials/sidebar.html`, un archivo sin usar de una
  versión anterior del diseño.
