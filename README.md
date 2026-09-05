# ML Ensemble Lab — Bagging vs Boosting

Aplicación Flask educativa que explica y compara **Bagging** (Random Forest) vs
**Boosting** (Gradient Boosting / AdaBoost) usando el dataset
`data2/heart_disease.csv` (10.000 filas, clasificación de riesgo cardíaco y
regresión de variables clínicas).

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
app.py                 # rutas Flask + API JSON
ml/pipeline.py          # carga de datos, entrenamiento, métricas y gráficas
templates/               # páginas (Overview/Boosting, Bagging, Comparative, Demo)
static/plots/            # gráficas generadas (se sobreescriben en cada entrenamiento)
data2/heart_disease.csv  # dataset usado por la app
data/                    # datasets adicionales, no usados por app.py actualmente
notebooks/               # notebook exploratorio (no requerido para correr la app)
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
`target` (solo en regresión: `Cholesterol Level`, `BMI`,
`Triglyceride Level` o `Blood Pressure`).

## Nota importante sobre los resultados

`heart_disease.csv` es un dataset **sintético** conocido (Kaggle) cuyas
variables tienen muy poca o nula relación estadística real con
`Heart Disease Status`. Por eso, aunque el pipeline funciona correctamente,
vas a ver:

- Exactitud (`accuracy`) cercana al 80%, que es simplemente la proporción de
  la clase mayoritaria ("No") — no señal real aprendida.
- `ROC AUC` rondando 0.48–0.52 (equivalente a adivinar al azar).
- `R²` de la regresión cercano a 0 o negativo.

Esto **no es un bug**: es una limitación real y documentada del dataset, y de
hecho es un buen punto de discusión pedagógico ("¿cómo se ve un modelo sin
señal? ¿por qué el accuracy engaña en datasets desbalanceados?"). Si quieres
resultados con más diferencia entre Bagging y Boosting para fines
demostrativos, reemplaza `data2/heart_disease.csv` por un dataset con señal
real (por ejemplo, el UCI Heart Disease clásico) manteniendo las mismas
columnas objetivo, o ajusta `ml/pipeline.py` para apuntar a otro archivo.

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
