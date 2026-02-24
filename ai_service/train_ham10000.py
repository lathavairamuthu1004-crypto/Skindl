import argparse
import json
import os
from pathlib import Path

import numpy as np


HAM_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train an EfficientNetB0 model on HAM10000.")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing HAM10000 images and HAM10000_metadata.csv (can be nested).",
    )
    p.add_argument(
        "--metadata",
        default=None,
        help="Optional path to HAM10000_metadata.csv. If omitted, searched under --data-dir.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (default: 224).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Head-training epochs with frozen backbone (default: 10).",
    )
    p.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Fine-tuning epochs with unfrozen backbone (default: 5).",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--out-model",
        default=str(Path("models") / "ham_efficientnetb0.keras"),
        help="Output model path (default: models/ham_efficientnetb0.keras).",
    )
    p.add_argument(
        "--out-labels",
        default=str(Path("models") / "ham_label_map.json"),
        help="Output labels JSON path (default: models/ham_label_map.json).",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    # Heavy imports here so `python app.py` doesn't require training deps.
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import tensorflow as tf

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"--data-dir does not exist: {data_dir}")

    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        matches = list(data_dir.rglob("HAM10000_metadata.csv"))
        if not matches:
            raise SystemExit("Could not find HAM10000_metadata.csv under --data-dir. Pass --metadata explicitly.")
        metadata_path = matches[0]

    df = pd.read_csv(metadata_path)
    if "image_id" not in df.columns or "dx" not in df.columns:
        raise SystemExit("metadata CSV must contain columns: image_id, dx")

    # Index image files by stem (image_id).
    img_paths = {}
    for p in data_dir.rglob("*.jpg"):
        img_paths[p.stem] = p

    df = df[df["dx"].isin(HAM_LABELS)].copy()
    df["path"] = df["image_id"].map(lambda x: str(img_paths.get(x, "")))
    df = df[df["path"].str.len() > 0].reset_index(drop=True)

    if len(df) == 0:
        raise SystemExit("No training rows found after matching metadata labels to image files.")

    label_to_index = {k: i for i, k in enumerate(HAM_LABELS)}
    y = df["dx"].map(label_to_index).astype(int).to_numpy()
    x = df["path"].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y,
    )

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(HAM_LABELS)),
        y=y_train,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}

    img_size = args.img_size
    batch_size = args.batch_size

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(min(len(x_train), 8192), seed=args.seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # Lightweight augmentation that doesn't distort lesions too much.
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
        include_top=False,
        weights="imagenet",
        input_tensor=x_in,
    )
    base.trainable = False

    x_out = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x_out = tf.keras.layers.Dropout(0.25)(x_out)
    outputs = tf.keras.layers.Dense(len(HAM_LABELS), activation="softmax")(x_out)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    out_model_path = Path(args.out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_model_path),
            monitor="val_acc",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Fine-tune: unfreeze backbone
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

    # Ensure final save and label map are written
    model.save(str(out_model_path))
    out_labels_path = Path(args.out_labels)
    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_path.write_text(json.dumps(HAM_LABELS, indent=2), encoding="utf-8")

    print(f"Saved model: {out_model_path}")
    print(f"Saved label map: {out_labels_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
