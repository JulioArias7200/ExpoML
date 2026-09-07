"""
ML Pipeline para EXPOULTI - Bagging vs Boosting
Dataset: UCI Heart Disease (Cleveland + Hungría + Suiza + Long Beach VA)
  data2/heart_disease/processed.cleveland.data
  data2/heart_disease/processed.hungarian.data
  data2/heart_disease/processed.switzerland.data
  data2/heart_disease/processed.va.data
- Clasificación: Heart Disease Status (Yes/No), derivado de la columna "num"
- Regresión: Cholesterol Level (alternativa: Resting Blood Pressure)
Usa solo scikit-learn / pandas / matplotlib / seaborn

Nota sobre el dataset: son los 4 archivos "processed.*.data" del repositorio
UCI Heart Disease. Cada uno trae las mismas 14 columnas ya seleccionadas por
los autores originales, con valores faltantes marcados como "?". Se combinan
los 4 centros en un solo dataset (920 filas) para tener más datos que usando
solo Cleveland (303 filas), a costa de más valores faltantes en las columnas
que no todos los centros midieron (ver comentarios en load_data).
"""

import os
from functools import lru_cache
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data2", "heart_disease")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Los 4 archivos "processed" del repositorio UCI Heart Disease (14 columnas,
# sin cabecera, valores faltantes como "?")
RAW_FILES = (
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
)
RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Renombramos a nombres legibles: así los gráficos de importancia de
# variables y las tablas de la demo se entienden sin tener que consultar
# el diccionario de datos original (heart-disease.names).
COLUMN_RENAME = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol Level",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels Colored",
    "thal": "Thalassemia",
}

# Códigos numéricos -> categorías legibles, según heart-disease.names.
# Convertirlos a texto (en vez de dejarlos como 1.0/2.0/3.0) hace que el
# OneHotEncoder los trate como categorías reales y no como una escala
# continua, y que las importancias de features salgan legibles.
SEX_MAP = {1: "Male", 0: "Female"}
CP_MAP = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
FBS_MAP = {1: "Yes", 0: "No"}
RESTECG_MAP = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
EXANG_MAP = {1: "Yes", 0: "No"}
SLOPE_MAP = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_MAP = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}

# Columnas
TARGET_CLASS = "Heart Disease Status"
TARGET_REG = "Max Heart Rate"       # principal regresión (fuerte correlación con Age y esfuerzo)
TARGET_REG_ALT = "Age"
CATEGORICAL_HINT = ["Sex", "Chest Pain Type", "Fasting Blood Sugar", "Resting ECG",
                     "Exercise Induced Angina", "ST Slope", "Thalassemia"]
NUMERIC_HINT = ["Age", "Resting Blood Pressure", "Cholesterol Level", "Max Heart Rate",
                "ST Depression", "Major Vessels Colored"]


@lru_cache(maxsize=4)
def _read_raw_cached(dir_path, files):
    # cachea la lectura/combinación de los 4 archivos: el demo reentrena
    # varias veces por sesión y volver a leer y unir los CSV cada vez era
    # el mayor costo de tiempo
    frames = []
    for fname in files:
        path = os.path.join(dir_path, fname)
        df = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
        df["Source"] = fname.replace("processed.", "").replace(".data", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_data(dir_path=DATA_DIR):
    # devolvemos una copia para que nadie mute el dataframe cacheado
    df = _read_raw_cached(dir_path, RAW_FILES).copy()

    # chol = 0 no es un colesterol real (sería incompatible con la vida): es
    # el valor centinela que usaron Suiza (123/123 filas) y Long Beach VA
    # (49/200 filas) para "no medido". Si no lo tratamos como NaN, el modelo
    # de regresión aprendería un patrón falso en vez de imputar de verdad.
    df.loc[df["chol"] == 0, "chol"] = np.nan

    # Mapear códigos numéricos a categorías legibles
    df["sex"] = df["sex"].map(SEX_MAP)
    df["cp"] = df["cp"].map(CP_MAP)
    df["fbs"] = df["fbs"].map(FBS_MAP)
    df["restecg"] = df["restecg"].map(RESTECG_MAP)
    df["exang"] = df["exang"].map(EXANG_MAP)
    df["slope"] = df["slope"].map(SLOPE_MAP)
    df["thal"] = df["thal"].map(THAL_MAP)

    # Target de clasificación: "num" original mide severidad angiográfica
    # (0 = sin obstrucción significativa, 1-4 = distintos grados). Igual que
    # en la versión anterior de este pipeline, la tarea de clasificación es
    # binaria: ¿hay evidencia de enfermedad cardíaca o no?
    df[TARGET_CLASS] = np.where(df["num"] > 0, "Yes", "No")
    df = df.drop(columns=["num"])

    # "Source" (de qué centro médico viene la fila) no se usa como feature:
    # es metadata de recolección, no un dato clínico del paciente, y además
    # se correlaciona demasiado con los valores faltantes (ver nota de chol).
    df = df.drop(columns=["Source"])

    df = df.rename(columns=COLUMN_RENAME)
    return df


def build_preprocessor(df, target_col, numeric_cols=None, categorical_cols=None):
    if numeric_cols is None or categorical_cols is None:
        # detectar automáticamente según dataset sin target
        all_cols = [c for c in df.columns if c != target_col]
        # num = dtype float/int
        numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in all_cols if c not in numeric_cols]
        # remover target de regresión si quedó
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(transformers=[
        ("num", numeric_tf, numeric_cols),
        ("cat", categorical_tf, categorical_cols)
    ])
    return pre, numeric_cols, categorical_cols


def split_classification(df, test_size=0.2, random_state=42):
    # mapear target a binario
    df = df.copy()
    df[TARGET_CLASS] = df[TARGET_CLASS].map({"Yes": 1, "No": 0})
    # eliminar filas donde target es NaN (no hay)
    df = df.dropna(subset=[TARGET_CLASS])
    X = df.drop(columns=[TARGET_CLASS])
    y = df[TARGET_CLASS].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_regression(df, target_col=TARGET_REG, test_size=0.2, random_state=42):
    df = df.copy()
    # eliminar target NaN (incluye los "colesterol 0" reconvertidos arriba)
    df = df.dropna(subset=[target_col])
    # Para regresión, evitar leakage: quitar Heart Disease Status del X
    drop_cols = [target_col]
    # opcional: si target es Cholesterol, también quitar Heart Disease Status para no filtrar info clínica futura
    if TARGET_CLASS in df.columns:
        drop_cols.append(TARGET_CLASS)
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------- ENTRENAMIENTO CLASIFICACIÓN ----------
def train_classification(n_estimators=100, learning_rate=0.1, random_state=42):
    df = load_data()
    X_train, X_test, y_train, y_test = split_classification(df, random_state=random_state)
    pre, num_cols, cat_cols = build_preprocessor(pd.concat([X_train, X_test]), TARGET_CLASS)

    # Modelos: Bagging (Random Forest) vs Boosting (GradientBoosting + AdaBoost)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    models = {"Bagging_RF": rf, "Boosting_GB": gb, "Boosting_Ada": ada}
    results = {}
    fitted = {}

    # Al combinar los 4 centros el desbalance de clases es más leve que en el
    # dataset anterior, pero igual conviene balancear para que la comparación
    # Bagging vs Boosting sea justa: RF ya balancea internamente vía
    # class_weight="balanced", GradientBoosting/AdaBoost no aceptan ese
    # parámetro, así que se replica el mismo balanceo con sample_weight.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", pre), ("classifier", clf)])
        if name == "Bagging_RF":
            pipe.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc,
                         "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba, "pipe": pipe}
        fitted[name] = pipe

    # guardar plots
    plot_paths = {}
    plot_paths["confusion_rf"] = plot_confusion(results["Bagging_RF"]["y_test"], results["Bagging_RF"]["y_pred"], "Bagging (RF) - Clasificación", "confusion_bagging.png")
    plot_paths["confusion_gb"] = plot_confusion(results["Boosting_GB"]["y_test"], results["Boosting_GB"]["y_pred"], "Boosting (GB) - Clasificación", "confusion_boosting.png")
    plot_paths["roc"] = plot_roc_curves(results, "roc_comparison.png")
    plot_paths["feat_rf"] = plot_feature_importance(fitted["Bagging_RF"], num_cols, cat_cols, "Bagging RF - Importancia", "feat_rf.png")
    plot_paths["feat_gb"] = plot_feature_importance(fitted["Boosting_GB"], num_cols, cat_cols, "Boosting GB - Importancia", "feat_gb.png")
    # NUEVOS plots prioritarios
    plot_paths["pr_curve"] = plot_pr_curves(results, "pr_curve.png")
    plot_paths["prob_dist"] = plot_prob_distribution(results, "prob_dist.png")
    plot_paths["threshold_curve"] = plot_threshold_curves(results, "threshold_curve.png")
    plot_paths["perm_importance"] = plot_permutation_importance(fitted["Bagging_RF"], X_test, y_test, num_cols, cat_cols, "perm_importance.png")
    plot_paths["corr_matrix"] = plot_correlation_heatmap(df, "Matriz de Correlación Clínica", "class_corr_heatmap.png")
    # tabla métricas
    metrics = {k: {m: float(v) for m, v in res.items() if m in ["accuracy", "precision", "recall", "f1", "roc_auc"] and v is not None} for k, res in results.items()}

    return {"metrics": metrics, "plots": plot_paths, "numeric_cols": num_cols, "categorical_cols": cat_cols}


# ---------- ENTRENAMIENTO REGRESIÓN ----------
def train_regression(target_col=TARGET_REG, n_estimators=100, learning_rate=0.1, random_state=42):
    df = load_data()
    X_train, X_test, y_train, y_test = split_regression(df, target_col=target_col, random_state=random_state)
    pre, num_cols, cat_cols = build_preprocessor(pd.concat([X_train, X_test]), target_col)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    models = {"Bagging_RF": rf, "Boosting_GB": gb}
    results = {}
    fitted = {}
    for name, reg in models.items():
        pipe = Pipeline(steps=[("preprocessor", pre), ("regressor", reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        results[name] = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "y_test": y_test, "y_pred": y_pred, "pipe": pipe}
        fitted[name] = pipe

    plot_paths = {}
    plot_paths["scatter_rf"] = plot_regression_scatter(results["Bagging_RF"]["y_test"], results["Bagging_RF"]["y_pred"], f"Bagging RF - {target_col}", "reg_scatter_rf.png")
    plot_paths["scatter_gb"] = plot_regression_scatter(results["Boosting_GB"]["y_test"], results["Boosting_GB"]["y_pred"], f"Boosting GB - {target_col}", "reg_scatter_gb.png")
    plot_paths["residual_rf"] = plot_residuals(results["Bagging_RF"]["y_test"], results["Bagging_RF"]["y_pred"], "Bagging RF - Residuales", "reg_residual_rf.png")
    plot_paths["residual_gb"] = plot_residuals(results["Boosting_GB"]["y_test"], results["Boosting_GB"]["y_pred"], "Boosting GB - Residuales", "reg_residual_gb.png")
    plot_paths["feat_rf"] = plot_feature_importance(fitted["Bagging_RF"], num_cols, cat_cols, "Bagging RF - Importancia", f"reg_feat_rf_{target_col.replace(' ', '_')}.png", is_regressor=True)
    plot_paths["feat_gb"] = plot_feature_importance(fitted["Boosting_GB"], num_cols, cat_cols, "Boosting GB - Importancia", f"reg_feat_gb_{target_col.replace(' ', '_')}.png", is_regressor=True)
    plot_paths["corr_matrix"] = plot_correlation_heatmap(df, f"Matriz de Correlación ({target_col})", f"reg_corr_{target_col.replace(' ', '_')}.png")

    metrics = {k: {m: float(v) for m, v in res.items() if m in ["mae", "mse", "rmse", "r2"]} for k, res in results.items()}
    return {"metrics": metrics, "plots": plot_paths, "target": target_col}


# ---------- PLOTS ----------
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title, fontsize=11)
    plt.xlabel("Predicho"); plt.ylabel("Real")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150); plt.close()
    return f"plots/{filename}"


def plot_roc_curves(results, filename):
    plt.figure(figsize=(5, 4))
    for name, res in results.items():
        if res["y_proba"] is not None:
            RocCurveDisplay.from_predictions(res["y_test"], res["y_proba"], name=name, ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.title("Curvas ROC - Bagging vs Boosting")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150); plt.close()
    return f"plots/{filename}"


def plot_feature_importance(pipe, num_cols, cat_cols, title, filename, is_regressor=False, top_n=10):
    # extraer feature names después de one-hot
    try:
        pre = pipe.named_steps["preprocessor"]
        # nombres one-hot
        cat_enc = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_enc.get_feature_names_out(cat_cols).tolist() if len(cat_cols) > 0 else []
        feature_names = num_cols + cat_features
        est = pipe.named_steps["classifier"] if not is_regressor else pipe.named_steps["regressor"]
        importances = est.feature_importances_
        # ordenar
        idx = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in idx]
        top_imp = importances[idx]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=top_imp, y=top_names, hue=top_names, palette="viridis", legend=False)
        plt.title(title, fontsize=11)
        plt.xlabel("Importancia"); plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150); plt.close()
        return f"plots/{filename}"
    except Exception as e:
        # fallback: no plot
        print("feat imp error", e)
        return None


def plot_regression_scatter(y_true, y_pred, title, filename):
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10, color="#006a61")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=1)
    plt.title(title, fontsize=11)
    plt.xlabel("Real"); plt.ylabel("Predicho")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150); plt.close()
    return f"plots/{filename}"


def plot_residuals(y_true, y_pred, title, filename):
    res = y_true - y_pred
    plt.figure(figsize=(4, 4))
    plt.scatter(y_pred, res, alpha=0.4, s=10, color="#93000a")
    plt.axhline(0, color="black", lw=0.8, ls="--")
    plt.title(title, fontsize=11)
    plt.xlabel("Predicho"); plt.ylabel("Residual")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150); plt.close()
    return f"plots/{filename}"


def plot_correlation_heatmap(df, title, filename):
    """Genera matriz de correlación anotada para explicar con quién correlaciona cada variable"""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num_cols].corr()
        plt.figure(figsize=(5.5, 4.5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-0.45, vmax=0.45, square=True, linewidths=0.8,
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
        plt.title(title, fontsize=10, fontweight="bold", pad=8)
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        return f"plots/{filename}"
    except Exception as e:
        print("corr heatmap error", e)
        return None


# ---------- NUEVOS PLOTS PARA CLASIFICACIÓN DESBALANCEADA ----------
def plot_pr_curves(results, filename):
    """Curva Precision-Recall - mejor que ROC para clase minoritaria"""
    plt.figure(figsize=(5, 4))
    for name, res in results.items():
        if res["y_proba"] is not None:
            prec, rec, _ = precision_recall_curve(res["y_test"], res["y_proba"])
            ap = average_precision_score(res["y_test"], res["y_proba"])
            plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})", lw=2)
    # baseline = proporción de positivos
    baseline = results[list(results.keys())[0]]["y_test"].mean()
    plt.axhline(baseline, color="gray", ls="--", lw=1, label=f"Baseline ({baseline:.3f})")
    plt.xlabel("Recall (Sensibilidad)"); plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall")
    plt.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150); plt.close()
    return f"plots/{filename}"


def plot_prob_distribution(results, filename):
    """Distribución de probabilidades predichas por clase real - diagnostica AUC ~0.5.

    Un panel con una subgráfica por modelo, en una sola imagen (evita el bug
    de generar solo la figura del primer modelo y dejar figuras sin cerrar).
    """
    names_with_proba = [n for n, r in results.items() if r.get("y_proba") is not None]
    if not names_with_proba:
        return None

    fig, axes = plt.subplots(1, len(names_with_proba), figsize=(5 * len(names_with_proba), 4), squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names_with_proba):
        res = results[name]
        y_test = res["y_test"]
        y_proba = res["y_proba"]
        sns.kdeplot(x=y_proba[y_test == 0], fill=True, alpha=0.35, label="Clase 0 (No)", color="#4575b4", ax=ax)
        sns.kdeplot(x=y_proba[y_test == 1], fill=True, alpha=0.35, label="Clase 1 (Yes)", color="#d73027", ax=ax)
        ax.axvline(0.5, color="black", ls="--", lw=1, label="Umbral 0.5")
        ax.set_xlabel("Probabilidad predicha de Clase 1")
        ax.set_ylabel("Densidad")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Distribución de probabilidades por modelo", fontsize=12)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return f"plots/{filename}"


def plot_threshold_curves(results, filename):
    """Métricas vs umbral de decisión - elige corte óptimo"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    thresholds = np.arange(0.05, 0.96, 0.05)
    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        if res["y_proba"] is not None:
            y_test = res["y_test"]
            y_proba = res["y_proba"]
            precs, recs, f1s = [], [], []
            for th in thresholds:
                y_pred_th = (y_proba >= th).astype(int)
                precs.append(precision_score(y_test, y_pred_th, zero_division=0))
                recs.append(recall_score(y_test, y_pred_th, zero_division=0))
                f1s.append(f1_score(y_test, y_pred_th, zero_division=0))
            plt.plot(thresholds, precs, '--', label=f"{name} Precision", alpha=0.7)
            plt.plot(thresholds, recs, '-.', label=f"{name} Recall", alpha=0.7)
            plt.plot(thresholds, f1s, '-', label=f"{name} F1", lw=2)
    plt.axvline(0.5, color="gray", ls=":", lw=1, label="Umbral 0.5")
    plt.xlabel("Umbral de decisión")
    plt.ylabel("Métrica")
    plt.title("Métricas vs Umbral")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150); plt.close()
    return f"plots/{filename}"


def plot_permutation_importance(pipe, X_test, y_test, num_cols, cat_cols, filename, scoring="average_precision", n_repeats=10):
    """Importancia por permutación en test set - más robusta que feature_importances_"""
    try:
        result = permutation_importance(pipe, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=42, n_jobs=-1)
        # nombres features
        pre = pipe.named_steps["preprocessor"]
        cat_enc = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_enc.get_feature_names_out(cat_cols).tolist() if len(cat_cols) > 0 else []
        feature_names = num_cols + cat_features

        imp_mean = result.importances_mean
        idx = np.argsort(imp_mean)[::-1][:15]
        top_names = [feature_names[i] for i in idx]
        top_imp = imp_mean[idx]

        plt.figure(figsize=(6, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_names)))
        bars = plt.barh(range(len(top_names)), top_imp[::-1], color=colors[::-1], edgecolor='white')
        plt.yticks(range(len(top_names)), top_names[::-1], fontsize=9)
        plt.xlabel(f"Disminución media en {scoring}")
        plt.title("Importancia por Permutación (Test Set)")
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150); plt.close()
        return f"plots/{filename}"
    except Exception as e:
        print("perm importance error", e)
        return None


def dataset_summary():
    df = load_data()
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing": df.isnull().sum().to_dict(),
        "class_balance": df[TARGET_CLASS].value_counts().to_dict(),
        "numeric_describe": df.describe().to_dict()
    }
    return summary