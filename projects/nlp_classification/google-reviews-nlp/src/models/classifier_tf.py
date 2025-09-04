# src/models/classifier_tf.py
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


# Project-wide label convention (must match your heuristic labels)
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}


@dataclass
class TrainConfig:
    """
    Training configuration. Update defaults or override via CLI.
    """
    csv_path: Path = Path("data/processed/reviews_labeled.csv")
    text_col: str = "message"
    label_col: str = "label"
    out_dir: Path = Path("artifacts/models/keras")

    # Model/training hyperparameters
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42

    # Vectorization
    num_words: int = 20000
    seq_len: int = 200

    # Splits
    val_size: float = 0.15
    test_size: float = 0.15

    # Imbalanced data
    class_weight: bool = True


# ---------- data helpers ----------

def _clean_text_series(s: pd.Series) -> pd.Series:
    """Lightweight text cleaning: remove links/mentions/hashtags, normalize, trim."""
    s = (
        s.astype(str)
        .str.replace(r"http\S+", " ", regex=True)
        .str.replace(r"@\w+", " ", regex=True)
        .str.replace(r"#[\w-]+", " ", regex=True)
        .str.normalize("NFKC")
        .str.strip()
    )
    return s


def _stratified_split(
    df: pd.DataFrame, label_col: str, val_size: float, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split preserving class ratios."""
    from sklearn.model_selection import train_test_split

    df_lab = df.dropna(subset=[label_col]).copy()
    train_df, tmp_df = train_test_split(
        df_lab,
        test_size=(val_size + test_size),
        stratify=df_lab[label_col],
        random_state=seed,
    )
    rel = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        tmp_df, test_size=(1 - rel), stratify=tmp_df[label_col], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _df_to_tfds(
    df: pd.DataFrame, text_col: str, label_col: str, batch: int, shuffle: bool, seed: int
) -> tf.data.Dataset:
    """Convert a pandas DataFrame into a tf.data.Dataset."""
    y = df[label_col].map(LABEL_TO_ID).astype("Int64")
    x = df[text_col].astype(str).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------- model ----------

def _build_model(num_words: int, seq_len: int, lr: float) -> Tuple[tf.keras.Model, layers.TextVectorization]:
    """
    Build a simple BiLSTM classifier:
      TextVectorization -> Embedding -> BiLSTM -> Dense -> Softmax
    """
    text_in = layers.Input(shape=(1,), dtype=tf.string, name="text")
    vectorizer = layers.TextVectorization(
        max_tokens=num_words,
        output_sequence_length=seq_len,
        name="vectorizer"
    )
    x = vectorizer(text_in)
    x = layers.Embedding(num_words, 128, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(LABELS), activation="softmax", name="probs")(x)

    model = tf.keras.Model(text_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, vectorizer


# ---------- training API ----------

class SentimentClassifierTF:
    """
    Inference wrapper for the saved Keras model.
    """
    def __init__(self, model_path: Path = Path("artifacts/models/keras/model_full")):
        self.model_path = model_path
        self.model: Optional[tf.keras.Model] = None

    def load(self):
        """Load a previously saved model from disk."""
        self.model = tf.keras.models.load_model(self.model_path)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Return class probabilities for a list of texts."""
        assert self.model is not None, "Model not loaded. Call .load() first."
        texts = np.array(list(texts), dtype=object)
        probs = self.model.predict(texts, verbose=0)
        return probs

    def predict(self, texts: Iterable[str]) -> List[str]:
        """Return predicted class labels for a list of texts."""
        probs = self.predict_proba(texts)
        preds = probs.argmax(axis=1)
        return [ID_TO_LABEL[int(i)] for i in preds]


def train_and_evaluate(cfg: TrainConfig = TrainConfig()) -> dict:
    """
    Train, evaluate and save model + quick metrics.
    - Saves full model (SavedModel) under cfg.out_dir / 'model_full'
    - Saves quick metrics JSON under cfg.out_dir / 'train_metrics.json'
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cfg.out_dir / "model_full.keras"
    
    

    tf.keras.utils.set_random_seed(cfg.seed)

    # Load dataset
    df = pd.read_csv(cfg.csv_path)
    if cfg.text_col not in df.columns:
        raise ValueError(f"Text column '{cfg.text_col}' not found in {cfg.csv_path}")
    if cfg.label_col not in df.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found in {cfg.csv_path}")

    # Filter to known labels and clean text
    df = df[df[cfg.label_col].isin(LABELS)].copy()
    df[cfg.text_col] = _clean_text_series(df[cfg.text_col])

    # Splits
    train_df, val_df, test_df = _stratified_split(df, cfg.label_col, cfg.val_size, cfg.test_size, cfg.seed)

    # Datasets
    train_ds = _df_to_tfds(train_df, cfg.text_col, cfg.label_col, cfg.batch_size, True, cfg.seed)
    val_ds   = _df_to_tfds(val_df,   cfg.text_col, cfg.label_col, cfg.batch_size, False, cfg.seed)
    test_ds  = _df_to_tfds(test_df,  cfg.text_col, cfg.label_col, cfg.batch_size, False, cfg.seed)

    # Model
    model, vectorizer = _build_model(cfg.num_words, cfg.seq_len, cfg.lr)
    # Adapt vectorizer on training texts
    vectorizer.adapt(train_df[cfg.text_col].values)

    # Optional class weights for imbalance
    cw = None
    if cfg.class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        y_train = train_df[cfg.label_col].map(LABEL_TO_ID).to_numpy()
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(LABELS)),
            y=y_train
        )
        cw = {i: float(w) for i, w in enumerate(weights)}

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(cfg.out_dir / "weights.keras", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # Save model (SavedModel directory)
    model.save(model_dir)

    # Save quick metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "class_weight": cw,
        "labels": LABELS,
    }
    (cfg.out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


# ---------- CLI ----------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a TensorFlow/Keras BiLSTM sentiment classifier.")
    p.add_argument("--csv", dest="csv_path", default="data/processed/reviews_labeled.csv", help="Input CSV path")
    p.add_argument("--text-col", dest="text_col", default="message", help="Text column name")
    p.add_argument("--label-col", dest="label_col", default="label", help="Label column name")
    p.add_argument("--out", dest="out_dir", default="artifacts/models/keras", help="Output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-words", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-class-weight", action="store_true", help="Disable class weighting")
    return p


def main():
    args = _build_argparser().parse_args()
    cfg = TrainConfig(
        csv_path=Path(args.csv_path),
        text_col=args.text_col,
        label_col=args.label_col,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_words=args.num_words,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=args.seed,
        class_weight=not args.no_class_weight,
    )
    metrics = train_and_evaluate(cfg)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()