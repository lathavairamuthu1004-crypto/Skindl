"""
Train a 9-class skin disease model (Acne, Eczema, Psoriasis, Melanoma, BCC, SCC, Melasma, Rosacea, Healthy).
Supports HAM10000 + multi-disease folder. Use --fast for quicker training.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# All 9 app conditions
CLASSES = [
    "Acne Vulgaris",
    "Atopic Dermatitis (Eczema)",
    "Psoriasis",
    "Malignant Melanoma",
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Melasma",
    "Rosacea",
    "Healthy Skin",
]

# HAM10000 dx -> our class
HAM_TO_CLASS = {
    "mel": "Malignant Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Squamous Cell Carcinoma",
    "nv": "Healthy Skin",
    "bkl": "Healthy Skin",
    "df": "Healthy Skin",
    "vasc": "Healthy Skin",
}

# Mendeley dataset folder names -> our class
MENDELEY_TO_CLASS = {
    "acne": "Acne Vulgaris",
    "nail_psoriasis": "Psoriasis",
    "nail-psoriasis": "Psoriasis",
    "hyperpigmentation": "Melasma",
    "vitiligo": "Healthy Skin",
}

# Multi-disease folder names (lowercase) -> our class
MULTI_FOLDER_TO_CLASS = {
    "acne": "Acne Vulgaris",
    "acne_vulgaris": "Acne Vulgaris",
    "eczema": "Atopic Dermatitis (Eczema)",
    "atopic_dermatitis": "Atopic Dermatitis (Eczema)",
    "psoriasis": "Psoriasis",
    "melasma": "Melasma",
    "rosacea": "Rosacea",
    "healthy": "Healthy Skin",
    "healthy_skin": "Healthy Skin",
    "melanoma": "Malignant Melanoma",
    "malignant_melanoma": "Malignant Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "basal_cell_carcinoma": "Basal Cell Carcinoma",
    "scc": "Squamous Cell Carcinoma",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma",
}


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train 9-class skin disease model.")
    p.add_argument("--ham-dir", help="HAM10000 folder (images + HAM10000_metadata.csv).")
    p.add_argument(
        "--multi-dir",
        help="Folder with subdirs: acne, eczema, psoriasis, melasma, rosacea, healthy.",
    )
    p.add_argument(
        "--mendeley-dir",
        help="Mendeley dataset folder (acne, nail_psoriasis, hyperpigmentation, etc.).",
    )
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64, help="Larger = faster (default 64).")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fine-tune-epochs", type=int, default=3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 3 epochs, no fine-tune, batch 64.",
    )
    p.add_argument("--demo", action="store_true", help="Create tiny demo data and train (for testing).")
    p.add_argument("--out-model", default="models/skin_full.keras")
    p.add_argument("--out-labels", default="models/skin_label_map.json")
    return p


def collect_ham10000(ham_dir: Path):
    """Collect (path, class_index) from HAM10000."""
    import pandas as pd

    matches = list(ham_dir.rglob("HAM10000_metadata.csv"))
    if not matches:
        return []
    df = pd.read_csv(matches[0])
    if "image_id" not in df.columns or "dx" not in df.columns:
        return []

    img_paths = {p.stem: p for p in ham_dir.rglob("*.jpg")}
    rows = []
    for _, r in df.iterrows():
        dx = r["dx"]
        if dx not in HAM_TO_CLASS:
            continue
        cls = HAM_TO_CLASS[dx]
        if cls not in CLASSES:
            continue
        path = img_paths.get(r["image_id"])
        if path and path.exists():
            rows.append((str(path), CLASSES.index(cls)))
    return rows


def collect_multi_dir(multi_dir: Path):
    """Collect (path, class_index) from folder structure."""
    rows = []
    for sub in multi_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower().replace(" ", "_").replace("-", "_")
        cls = MULTI_FOLDER_TO_CLASS.get(name)
        if cls is None:
            continue
        idx = CLASSES.index(cls)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in sub.glob(ext):
                rows.append((str(p), idx))
    return rows


def collect_mendeley(mendeley_dir: Path):
    """Collect from Mendeley-style folder names."""
    rows = []
    for sub in mendeley_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower().replace(" ", "_").replace("-", "_")
        cls = MENDELEY_TO_CLASS.get(name)
        if cls is None:
            continue
        idx = CLASSES.index(cls)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in sub.glob(ext):
                rows.append((str(p), idx))
    return rows


DEMO_FOLDERS = [
    "acne", "eczema", "psoriasis", "melasma", "rosacea", "healthy",
    "melanoma", "bcc", "scc",
]


def _create_demo_data(out_dir: Path):
    """Create minimal demo images for pipeline testing (uses PIL, no TensorFlow)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for folder in DEMO_FOLDERS:
        sub = out_dir / folder
        sub.mkdir(exist_ok=True)
        for j in range(4):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(sub / f"demo_{j}.jpg")
    print(f"Demo data: {out_dir}")


def main():
    args = build_arg_parser().parse_args()

    if args.fast:
        args.epochs = 3
        args.fine_tune_epochs = 0
        args.batch_size = 64

    if args.demo:
        demo_dir = Path("data") / "demo_multi"
        _create_demo_data(demo_dir)
        args.multi_dir = str(demo_dir)

    all_rows = []

    if args.ham_dir:
        ham_path = Path(args.ham_dir)
        if ham_path.exists():
            rows = collect_ham10000(ham_path)
            all_rows.extend(rows)
            print(f"HAM10000: {len(rows)} images")

    if args.multi_dir:
        multi_path = Path(args.multi_dir)
        if multi_path.exists():
            rows = collect_multi_dir(multi_path)
            all_rows.extend(rows)
            print(f"Multi-dir: {len(rows)} images")

    if args.mendeley_dir:
        men_path = Path(args.mendeley_dir)
        if men_path.exists():
            rows = collect_mendeley(men_path)
            all_rows.extend(rows)
            print(f"Mendeley: {len(rows)} images")

    if not all_rows:
        print(
            "No data. Use --ham-dir, --multi-dir, and/or --mendeley-dir.\n"
            "Example: python train_full.py --ham-dir data/HAM10000 --multi-dir data/skin_diseases"
        )
        raise SystemExit(1)

    paths = np.array([r[0] for r in all_rows])
    labels = np.array([r[1] for r in all_rows])
    n_classes = len(CLASSES)

    # Filter to classes present
    present = set(labels)
    if len(present) < 2:
        print("Need at least 2 classes. Add more data sources.")
        raise SystemExit(1)

    try:
        import tensorflow as tf
    except ImportError as e:
        print(
            "TensorFlow is required for training. Install with:\n"
            "  pip install tensorflow\n"
            "If you see 'No matching distribution found', your Python may be too new.\n"
            "TensorFlow supports Python 3.9-3.12. Try:\n"
            "  py -3.11 -m venv .venv\n"
            "  .venv\\Scripts\\activate\n"
            "  pip install -r requirements-train.txt\n"
            f"Original error: {e}"
        )
        raise SystemExit(1)

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    # Stratification requires at least one sample per class in the test set.
    # If test_size < n_classes, scikit-learn will throw a ValueError.
    n_present = len(present)
    test_size = args.val_split
    if isinstance(test_size, float):
        test_count = int(len(paths) * test_size)
    else:
        test_count = int(test_size)

    if test_count < n_present:
        print(f"--- INFO: Dataset too small for default {args.val_split} split with stratification.")
        print(f"--- INFO: Adjusting test count from {test_count} to {n_present} (number of classes).")
        test_count = n_present

    if test_count >= len(paths):
        print("--- WARNING: Dataset extremely small. Disabling stratification.")
        x_train, x_val, y_train, y_val = train_test_split(
            paths, labels, test_size=args.val_split, random_state=args.seed
        )
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            paths, labels, test_size=test_count, random_state=args.seed, stratify=labels
        )

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}

    img_size = args.img_size
    batch_size = args.batch_size

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = img[..., :3]  # ensure 3 channels (PNG may have alpha)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(x_train), 8192), seed=args.seed, reshuffle_each_iteration=True)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Mixed precision for faster GPU training
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )
    x_in = aug(inputs)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x_in
    )
    base.trainable = False
    x_out = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x_out = tf.keras.layers.Dropout(0.25)(x_out)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", dtype="float32")(x_out)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_model), monitor="val_acc", save_best_only=True
        ),
    ]

    print(f"Training {len(x_train)} samples, {len(present)} classes...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        base.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weight,
        )

    model.save(str(out_model))
    out_labels = Path(args.out_labels)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_labels.write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    print(f"Saved: {out_model}")
    print(f"Labels: {out_labels}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
