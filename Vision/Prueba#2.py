
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, transform, exposure
from skimage.util import img_as_ubyte
from skimage.feature import (
    hog,
    local_binary_pattern,
    graycomatrix,
    graycoprops
)

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

dataset_dir = r"C:\Users\espar\Documents\Proyecto\Vision\Vision\Prueba #2\data"

output_dir = os.path.join(
    r"C:\Users\espar\Documents\Proyecto\Vision\Vision\Prueba #2\data\features_Corns\visualizador_descriptores"
)
os.makedirs(output_dir, exist_ok=True)

classes = ["Common_Rust", "Corn_Blight", "Healthy","Gray_Leaf_Spot"]

extensions = [
    "*.jpg", "*.jpeg", "*.png",
    "*.bmp", "*.tif", "*.tiff"
]

image_size = (128, 128)

# Configuración HOG
HOG_ORIENTATIONS = 12
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Configuraciones LBP multiescala
LBP_SCALES = [
    (1, 8),
    (2, 16),
    (4, 32)
]


# ============================================================
# 2. SELECCIONAR UNA IMAGEN POR CLASE
# ============================================================

def find_one_image(class_name):
    folder = os.path.join(dataset_dir, class_name)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"No existe la carpeta: {folder}"
        )

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(folder, "**", ext),
                recursive=True
            )
        )

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {folder}"
        )

    # Selección reproducible de una sola imagen
    return files[0]


# ============================================================
# 3. PREPARAR IMAGEN Y DESCRIPTORES
# ============================================================

def load_image(image_path):
    image = io.imread(image_path)

    # Convertir a RGB si la imagen tiene canal alfa
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    # Preparar imagen en escala de grises para descriptores
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32)

        if gray.max() > 1:
            gray = gray / 255.0

    gray = transform.resize(
        gray,
        image_size,
        anti_aliasing=True
    )

    return image, gray


# ============================================================
# 4. CREAR COMPARACIÓN EN PARALELO
# ============================================================

n_rows = 2 + len(LBP_SCALES) + 1
n_cols = len(classes)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(10, 3.0 * n_rows),
    squeeze=False
)

fig.suptitle(
    "Comparación de descriptores HOG, LBP multiescala y GLCM",
    fontsize=17,
    fontweight="bold",
    y=0.995
)

for col, class_name in enumerate(classes):

    image_path = find_one_image(class_name)
    original, gray = load_image(image_path)

    print(f"{class_name}: {image_path}")

    # --------------------------------------------------------
    # FILA 1: IMAGEN ORIGINAL
    # --------------------------------------------------------

    ax = axes[0, col]

    if original.ndim == 2:
        ax.imshow(original, cmap="gray")
    else:
        ax.imshow(original)

    ax.set_title(class_name, fontweight="bold", fontsize=13)
    ax.axis("off")

    # --------------------------------------------------------
    # FILA 2: HOG
    # --------------------------------------------------------

    hog_features, hog_image = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True
    )

    hog_image = exposure.rescale_intensity(
        hog_image,
        in_range="image",
        out_range=(0, 1)
    )

    ax = axes[1, col]
    ax.imshow(hog_image, cmap="magma")
    ax.set_title(
        f"HOG (8×8, {HOG_ORIENTATIONS} orient.)\n"
        f"{len(hog_features)} características",
        fontsize=10
    )
    ax.axis("off")

    # --------------------------------------------------------
    # FILAS 3, 4 Y 5: LBP MULTIESCALA
    # --------------------------------------------------------

    gray_uint8 = img_as_ubyte(gray)

    for scale_idx, (radius, points) in enumerate(LBP_SCALES):

        lbp = local_binary_pattern(
            gray_uint8,
            P=points,
            R=radius,
            method="uniform"
        )

        ax = axes[2 + scale_idx, col]

        ax.imshow(
            lbp,
            cmap="gray",
            interpolation="nearest"
        )

        ax.set_title(
            f"LBP (R={radius}, P={points})",
            fontsize=10
        )
        ax.axis("off")

    # --------------------------------------------------------
    # ÚLTIMA FILA: GLCM
    # --------------------------------------------------------

    # Cuantizar la imagen en 8 niveles de gris
    quantized = (gray_uint8 // 32).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    # Matriz GLCM real de 8 × 8
    matrix = glcm[:, :, 0, 0]

    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    ax = axes[-1, col]

    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title(
        "GLCM (8 niveles, distancia 1)",
        fontsize=10
    )
    ax.set_xlabel("Nivel vecino")
    ax.set_ylabel("Nivel de referencia")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))

    # Valores estadísticos debajo de la matriz
    properties_text = (
        f"Contraste: {contrast:.3f} | "
        f"Homogeneidad: {homogeneity:.3f}\n"
        f"Energía: {energy:.3f} | "
        f"Correlación: {correlation:.3f}"
    )

    ax.text(
        0.5, -0.25,
        properties_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8
    )

# Ajustar espacios
plt.tight_layout(rect=[0, 0.015, 1, 0.98], h_pad=1.5, w_pad=1.0)

output_path = os.path.join(
    output_dir,
    "comparacion_HOG_LBP_GLCM_CommonRust_CornBlight.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nVisualizador generado correctamente.")
print("Archivo:", output_path)
