from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier


DEFAULT_DATA_PATH = Path(r"D:\ADNI1_Merged_MRI_Metadata.csv")
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "adni_medbert_outputs"
DEFAULT_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_TARGET_COLUMN = "Group"
DEFAULT_GROUP_COLUMN = "Subject"
DEFAULT_ID_COLUMNS = ["Image Data ID", "Subject"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Med-BERT embeddings from ADNI tabular metadata and run classification."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--target-column", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--group-column", type=str, default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--reuse-embeddings-csv",
        type=Path,
        default=None,
        help="Skip embedding generation and use an existing embeddings CSV.",
    )
    parser.add_argument(
        "--include-columns",
        nargs="*",
        default=None,
        help="Optional explicit feature columns to serialize into text.",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=None,
        help="Optional extra columns to exclude from text serialization.",
    )
    return parser.parse_args()


def normalize_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(float(value))
    text = str(value).strip()
    return text if text else "unknown"


def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    include_columns: list[str] | None,
    exclude_columns: list[str] | None,
) -> list[str]:
    if include_columns:
        return [col for col in include_columns if col in df.columns and col != target_column]

    ignored = set(DEFAULT_ID_COLUMNS + [target_column])
    if exclude_columns:
        ignored.update(exclude_columns)
    return [col for col in df.columns if col not in ignored]


def build_text_prompt(row: pd.Series, feature_columns: Iterable[str]) -> str:
    fragments = []
    for column in feature_columns:
        fragments.append(f"{column.lower().replace('_', ' ')} is {normalize_value(row[column])}")
    return ". ".join(fragments) + "."


def load_and_prepare_dataframe(
    data_path: Path,
    target_column: str,
    include_columns: list[str] | None,
    exclude_columns: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {data_path}")

    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    feature_columns = infer_feature_columns(df, target_column, include_columns, exclude_columns)
    if not feature_columns:
        raise ValueError("No usable feature columns were found for text serialization.")

    df["text_prompt"] = df.apply(lambda row: build_text_prompt(row, feature_columns), axis=1)
    return df, feature_columns


def generate_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{model_name}' from the local Hugging Face cache. "
            "Connect to the internet once to download it, or reuse an existing embeddings CSV."
        ) from exc
    model.eval()

    all_embeddings: list[np.ndarray] = []
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def save_embeddings_csv(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_path: Path,
    group_column: str,
    target_column: str,
) -> None:
    embedding_columns = [f"medbert_{idx}" for idx in range(embeddings.shape[1])]
    export_df = pd.DataFrame(embeddings, columns=embedding_columns)
    if group_column in df.columns:
        export_df.insert(0, group_column, df[group_column].values)
    export_df.insert(1 if group_column in df.columns else 0, target_column, df[target_column].values)
    export_df.to_csv(output_path, index=False)


def load_embeddings_from_csv(
    embeddings_csv: Path,
    target_column: str,
    group_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    if not embeddings_csv.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_csv}")

    df = pd.read_csv(embeddings_csv)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {embeddings_csv}")

    embedding_columns = [col for col in df.columns if col.startswith("medbert_")]
    if not embedding_columns:
        raise ValueError(f"No 'medbert_' columns found in {embeddings_csv}")

    X = df[embedding_columns].to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column].astype(str))
    groups = (
        df[group_column].astype(str).to_numpy()
        if group_column in df.columns
        else np.array(["no_group"] * len(df))
    )
    return df, X, y, groups, label_encoder


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return train_idx, test_idx

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y))
    return train_idx, test_idx


def evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_encoder: LabelEncoder,
    random_state: int,
) -> list[dict]:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    estimators = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        early_stopping=True,
                        max_iter=300,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
        ),
    }

    results: list[dict] = []
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        results.append(
            {
                "classifier": name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    target_names=label_encoder.classes_,
                    output_dict=True,
                    zero_division=0,
                ),
            }
        )

    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_csv_path = args.output_dir / "medbert_embeddings.csv"
    embeddings_npy_path = args.output_dir / "medbert_embeddings.npy"
    prompts_csv_path = args.output_dir / "serialized_prompts.csv"
    metrics_json_path = args.output_dir / "classification_metrics.json"
    summary_csv_path = args.output_dir / "classification_summary.csv"

    if args.reuse_embeddings_csv is None:
        df, feature_columns = load_and_prepare_dataframe(
            data_path=args.data_path,
            target_column=args.target_column,
            include_columns=args.include_columns,
            exclude_columns=args.exclude_columns,
        )
        df[
            [col for col in [*DEFAULT_ID_COLUMNS, args.target_column, "text_prompt"] if col in df.columns]
        ].to_csv(prompts_csv_path, index=False)

        embeddings = generate_embeddings(
            texts=df["text_prompt"].tolist(),
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        np.save(embeddings_npy_path, embeddings)
        save_embeddings_csv(
            df=df,
            embeddings=embeddings,
            output_path=embeddings_csv_path,
            group_column=args.group_column,
            target_column=args.target_column,
        )
        print(f"Saved prompts to: {prompts_csv_path}")
        print(f"Saved embeddings matrix to: {embeddings_npy_path}")
        print(f"Saved embeddings table to: {embeddings_csv_path}")
        print(f"Serialized {len(feature_columns)} feature columns into text.")
    else:
        embeddings_csv_path = args.reuse_embeddings_csv
        print(f"Reusing embeddings from: {embeddings_csv_path}")

    _, X, y, groups, label_encoder = load_embeddings_from_csv(
        embeddings_csv=embeddings_csv_path,
        target_column=args.target_column,
        group_column=args.group_column,
    )
    train_idx, test_idx = make_split(
        X=X,
        y=y,
        groups=groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    results = evaluate_models(
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        label_encoder=label_encoder,
        random_state=args.random_state,
    )

    summary_df = pd.DataFrame(
        [{"classifier": item["classifier"], "accuracy": item["accuracy"]} for item in results]
    ).sort_values("accuracy", ascending=False)
    summary_df.to_csv(summary_csv_path, index=False)
    metrics_json_path.write_text(json.dumps(results, indent=2))

    print("\nClasses:", list(label_encoder.classes_))
    print("Train rows:", len(train_idx))
    print("Test rows:", len(test_idx))
    print("\nClassification summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_csv_path}")
    print(f"Saved detailed metrics to: {metrics_json_path}")


if __name__ == "__main__":
    main()
