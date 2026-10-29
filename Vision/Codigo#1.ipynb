import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================
# Cambia esta ruta para que apunte a la carpeta que contiene
# training, validation y test.
BASE_DIR = r"C:\Users\espar\Documents\Proyecto\Vision\Vision\Prueba #2\data"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")
OUTPUT_DIR = os.path.join(BASE_DIR, "features_Corns")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (128, 128)
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Nombres de las carpetas/clases del dataset.

EXPECTED_CLASSES = [
    "Common_Rust",
    "Corn_Blight",
    "Gray_Leaf_Spot",
    "Healthy",
]

# ============================================================
# 2. CARGA DE IMÁGENES
# ============================================================
def load_dataset(folder, expected_classes):
    """Carga las imágenes de las carpetas de clase y devuelve X, y."""
    images = []
    labels = []
    counts = {}

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"No se encontró la carpeta: {folder}\n"
            "Revisa BASE_DIR y los nombres de las carpetas."
        )

    available = set(os.listdir(folder))
    missing = [c for c in expected_classes if c not in available]
    if missing:
        raise FileNotFoundError(
            f"Faltan estas carpetas dentro de {folder}: {missing}\n"
            f"Carpetas encontradas: {sorted(available)}"
        )

    for class_name in expected_classes:
        class_dir = os.path.join(folder, class_name)
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(EXTENSIONS)
        )

        valid_count = 0
        for filename in files:
            path = os.path.join(class_dir, filename)
            image = cv2.imread(path)

            if image is None:
                print(f"Advertencia: no se pudo leer {path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            image = image.astype(np.float32) / 255.0

            images.append(image)
            labels.append(class_name)
            valid_count += 1

        counts[class_name] = valid_count

    if not images:
        raise ValueError(f"No se cargaron imágenes desde {folder}")

    print(f"\nImágenes cargadas desde: {folder}")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")

    return np.asarray(images, dtype=np.float32), np.asarray(labels), counts


print("Cargando conjunto de entrenamiento...")
X_train_img, y_train_text, train_counts = load_dataset(TRAIN_DIR, EXPECTED_CLASSES)

print("\nCargando conjunto de prueba...")
X_test_img, y_test_text, test_counts = load_dataset(TEST_DIR, EXPECTED_CLASSES)

# Codificación consistente: las mismas etiquetas numéricas para train y test.
label_encoder = LabelEncoder()
label_encoder.fit(EXPECTED_CLASSES)
y_train = label_encoder.transform(y_train_text)
y_test = label_encoder.transform(y_test_text)
class_names = list(label_encoder.classes_)

print("\nResumen del dataset:")
print(f"Entrenamiento: {X_train_img.shape}; etiquetas: {y_train.shape}")
print(f"Prueba:        {X_test_img.shape}; etiquetas: {y_test.shape}")
print(f"Número de clases: {len(class_names)}")

# ============================================================
# 3. EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================
def to_gray_uint8(img):
    """Entrada RGB float32 en [0,1] -> imagen gris uint8."""
    return cv2.cvtColor(
        np.clip(img * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

def extraer_hog(imagenes):
    features = []
    for img in imagenes:
        gray = to_gray_uint8(img)
        vector = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )
        features.append(vector)
    return np.asarray(features, dtype=np.float32)

def extraer_lbp(imagenes, P=8, R=1):
    features = []
    for img in imagenes:
        gray = to_gray_uint8(img)
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, P + 3),
            range=(0, P + 2)
        )
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        features.append(hist)
    return np.asarray(features, dtype=np.float32)

def extraer_glcm(imagenes, distances=(1, 3),
                 angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    features = []
    properties = [
        "contrast", "dissimilarity", "homogeneity",
        "energy", "correlation", "ASM"
    ]

    for img in imagenes:
        gray = to_gray_uint8(img)
        gray_quantized = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(
            gray_quantized,
            distances=list(distances),
            angles=list(angles),
            levels=64,
            symmetric=True,
            normed=True
        )

        vector = []
        for prop in properties:
            vector.extend(graycoprops(glcm, prop).ravel())
        features.append(vector)

    return np.asarray(features, dtype=np.float32)

# ============================================================
# 4. EXTRACCIÓN DE DESCRIPTORES
# ============================================================
print("\nExtrayendo HOG de entrenamiento y test...")
X_train_hog = extraer_hog(X_train_img)
X_test_hog = extraer_hog(X_test_img)

print("Extrayendo LBP de entrenamiento y test...")
X_train_lbp = extraer_lbp(X_train_img)
X_test_lbp = extraer_lbp(X_test_img)

print("Extrayendo GLCM de entrenamiento y test...")
X_train_glcm = extraer_glcm(X_train_img)
X_test_glcm = extraer_glcm(X_test_img)

print("\nDimensiones sin escalar:")
print(f"HOG:  train={X_train_hog.shape}, test={X_test_hog.shape}")
print(f"LBP:  train={X_train_lbp.shape}, test={X_test_lbp.shape}")
print(f"GLCM: train={X_train_glcm.shape}, test={X_test_glcm.shape}")

# Fusión de los tres vectores.
X_train_fusion = np.hstack((X_train_hog, X_train_lbp, X_train_glcm))
X_test_fusion = np.hstack((X_test_hog, X_test_lbp, X_test_glcm))

# ============================================================
# 5. ESCALADO Z-SCORE SIN FUGA DE DATOS
# ============================================================
# Cada scaler se ajusta SOLO con training; test solo se transforma.
feature_sets_raw = {
    "HOG": (X_train_hog, X_test_hog),
    "LBP": (X_train_lbp, X_test_lbp),
    "GLCM": (X_train_glcm, X_test_glcm),
    "FUSIÓN (HOG+LBP+GLCM)": (X_train_fusion, X_test_fusion),
}

feature_sets = {}
scalers = {}

for name, (Xtr, Xte) in feature_sets_raw.items():
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)
    feature_sets[name] = (Xtr_scaled, Xte_scaled)
    scalers[name] = scaler

    # Guardar características y etiquetas para poder reutilizarlas.
    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "_")
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"features_{safe_name}.npz"),
        X_train=Xtr_scaled,
        X_test=Xte_scaled,
        y_train=y_train,
        y_test=y_test,
        class_names=np.asarray(class_names)
    )

print(f"\nCaracterísticas guardadas en: {OUTPUT_DIR}")

# ============================================================
# 6. COMPARACIÓN DE CLASIFICADORES
# ============================================================
# No se usa validation. La selección preliminar se hace con
# validación cruzada dentro de training; test se reserva para
# la evaluación final comparativa.
models = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}

# Validación cruzada solo sobre training para una referencia de selección.
# Se usa un máximo de 5 folds y se comprueba que haya suficientes ejemplos
# por clase para poder dividir de forma estratificada.
min_class_count = int(np.bincount(y_train).min())
n_splits = min(5, min_class_count)

cv_rows = []
if n_splits >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\nValidación cruzada estratificada en training: {n_splits} folds")

    for feature_name, (Xtr, _) in feature_sets.items():
        for model_name, model in models.items():
            start = time.time()
            # El scaler ya se calculó con todo training; para una CV
            # estrictamente sin fuga, el escalado se hace dentro del pipeline.
            # Aquí se usa Pipeline para que cada fold ajuste su propio scaler
            # sobre el fold de entrenamiento.
            raw_train = feature_sets_raw[feature_name][0]
            pipeline = make_pipeline(StandardScaler(), model)
            scores = cross_val_score(
                pipeline, raw_train, y_train, cv=cv,
                scoring="f1_macro", n_jobs=None
            )
            cv_rows.append({
                "Descriptor": feature_name,
                "Modelo": model_name,
                "CV F1-macro media": scores.mean(),
                "CV F1-macro desviación": scores.std(),
                "Tiempo CV (s)": time.time() - start,
            })
else:
    print(
        "\nNo se puede hacer validación cruzada estratificada: "
        "alguna clase tiene menos de 2 imágenes en training."
    )

df_cv = pd.DataFrame(cv_rows)
if not df_cv.empty:
    print("\n" + "=" * 100)
    print("RESULTADOS DE VALIDACIÓN CRUZADA (SOLO TRAINING)")
    print("=" * 100)
    print(df_cv.sort_values("CV F1-macro media", ascending=False).to_string(index=False))

# Comparación final en test de las 12 combinaciones.
results = []
fitted_models = {}

print("\nEntrenando modelos y evaluando en test...")
for feature_name, (Xtr, Xte) in feature_sets.items():
    for model_name, model_template in models.items():
        model = model_template.__class__(**model_template.get_params())

        start = time.time()
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        elapsed = time.time() - start

        results.append({
            "Descriptor": feature_name,
            "Modelo": model_name,
            "Dimensiones": Xtr.shape[1],
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision macro": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall macro": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1 macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1 weighted": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Tiempo entrenamiento + predicción (s)": elapsed,
        })
        fitted_models[(feature_name, model_name)] = model

df_results = pd.DataFrame(results)

print("\n" + "=" * 120)
print("TABLA COMPARATIVA FINAL EN TEST (15 CLASES)")
print("=" * 120)
print(
    df_results.sort_values("F1 macro", ascending=False)
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

# Guardar tablas.
df_results.to_csv(
    os.path.join(OUTPUT_DIR, "comparacion_test.csv"), index=False
)
if not df_cv.empty:
    df_cv.to_csv(
        os.path.join(OUTPUT_DIR, "comparacion_validacion_cruzada.csv"),
        index=False
    )

# Gráfico F1 macro.
pivot_f1 = df_results.pivot(
    index="Descriptor", columns="Modelo", values="F1 macro"
)
ax = pivot_f1.plot(kind="bar", figsize=(12, 6), edgecolor="black")
ax.set_title("Comparación de clasificadores y descriptores (F1 macro)")
ax.set_ylabel("F1 macro")
ax.set_xlabel("Descriptor")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(title="Clasificador")
plt.tight_layout()
plt.show()

# ============================================================
# 7. REPORTE DETALLADO DEL MEJOR RESULTADO EN TEST
# ============================================================
best_row = df_results.sort_values("F1 macro", ascending=False).iloc[0]
best_key = (best_row["Descriptor"], best_row["Modelo"])
best_model = fitted_models[best_key]

X_train_best, X_test_best = feature_sets[best_key[0]]
y_test_pred = best_model.predict(X_test_best)

print("\nMejor combinación por F1 macro en test:")
print(best_row.to_string())

print("\nReporte por clase para esa combinación:")
print(classification_report(
    y_test,
    y_test_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(
    y_test, y_test_pred, labels=np.arange(len(class_names))
)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(13, 12))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
ax.set_title(f"Matriz de confusión: {best_key[0]} + {best_key[1]}")
plt.tight_layout()
plt.show()

print(
    "\nNota metodológica: la tabla de test sirve para el informe final, "
    "pero elegir el ganador mirando test introduce sesgo. Para una evaluación "
    "estricta, elige la combinación según CV F1-macro y evalúa esa única "
    "combinación en test una sola vez."
)
