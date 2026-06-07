from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_PATH = Path(r"D:\ADNI1_Merged_MRI_Metadata.csv")
OUTPUT_BASE = Path(tempfile.gettempdir()) / "adni_diagnosis_outputs"
REPORT_PATH = OUTPUT_BASE / "diagnosis_report.json"
PREDICTIONS_PATH = OUTPUT_BASE / "test_set_predictions.csv"
TARGET_COLUMN = "Group"
GROUP_COLUMN = "Subject"
RANDOM_STATE = 42

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN]).copy()
    return df


def encode_visit_month(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.lower().str.strip()
    return (
        cleaned.str.extract(r"(\d+)", expand=False).fillna("-1").astype(int)
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Visit" in df.columns:
        df["VisitMonth"] = encode_visit_month(df["Visit"])
    if "VISCODE" in df.columns:
        df["VISCODE_month"] = encode_visit_month(df["VISCODE"])
    if "VISCODE2" in df.columns:
        df["VISCODE2_month"] = encode_visit_month(df["VISCODE2"])

    if "Description" in df.columns:
        desc = df["Description"].fillna("").astype(str)
        df["Description_has_B1"] = desc.str.contains("B1 Correction", case=False).astype(int)
        df["Description_has_GradWarp"] = desc.str.contains("GradWarp", case=False).astype(int)
        df["Description_variant_count"] = desc.str.count(";")

    for date_col in ["Acq Date", "VISDATE", "USERDATE"]:
        if date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            df[f"{date_col}_year"] = parsed.dt.year
            df[f"{date_col}_month"] = parsed.dt.month

    return df


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = engineer_features(df)
    y = df[TARGET_COLUMN].copy()
    groups = df[GROUP_COLUMN].copy()

    drop_columns = {
        TARGET_COLUMN,
        "Image Data ID",
        "Subject",
        "PTID",
        "RID",
        "ID",
        "USERDATE2",
        "update_stamp",
        "Downloaded",
        "Modality",
        "Type",
        "Format",
        "PHASE",
    }

    X = df.drop(columns=[c for c in drop_columns if c in df.columns]).copy()

    low_signal = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if low_signal:
        X = X.drop(columns=low_signal)

    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().mean() >= 0.8:
                X[col] = converted

    return X, y, groups


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> list[tuple[str, Pipeline, bool]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=900,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "extra_trees",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=1200,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "rf_extra_trees_vote",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        VotingClassifier(
                            estimators=[
                                (
                                    "rf",
                                    RandomForestClassifier(
                                        n_estimators=900,
                                        min_samples_leaf=1,
                                        max_features="sqrt",
                                        class_weight="balanced_subsample",
                                        random_state=RANDOM_STATE,
                                        n_jobs=1,
                                    ),
                                ),
                                (
                                    "et",
                                    ExtraTreesClassifier(
                                        n_estimators=1200,
                                        min_samples_leaf=1,
                                        max_features="sqrt",
                                        class_weight="balanced",
                                        random_state=RANDOM_STATE,
                                        n_jobs=1,
                                    ),
                                ),
                            ],
                            voting="soft",
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "xgboost",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        XGBClassifier(
                            objective="multi:softmax",
                            num_class=3,
                            n_estimators=350,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_lambda=1.5,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                            eval_metric="mlogloss",
                        ),
                    ),
                ]
            ),
            True,
        ),
        (
            "lightgbm",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LGBMClassifier(
                            objective="multiclass",
                            num_class=3,
                            n_estimators=350,
                            learning_rate=0.05,
                            num_leaves=31,
                            max_depth=-1,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
            True,
        ),
    ]


def cross_validate_model(
    model: Pipeline,
    use_sample_weight: bool,
    encode_target: bool,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> dict:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[dict] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        y_train_fit = y_train
        y_valid_eval = y_valid

        fitted = clone(model)
        fit_kwargs = {}
        if encode_target:
            class_names = sorted(y_train.unique().tolist())
            label_to_int = {label: idx for idx, label in enumerate(class_names)}
            int_to_label = {idx: label for label, idx in label_to_int.items()}
            y_train_fit = y_train.map(label_to_int)
        if use_sample_weight:
            fit_kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", y_train_fit)
        fitted.fit(X_train, y_train_fit, **fit_kwargs)
        preds = fitted.predict(X_valid)
        if encode_target:
            preds = pd.Series(preds).map(int_to_label).to_numpy()

        fold_scores.append(
            {
                "fold": fold_idx,
                "accuracy": accuracy_score(y_valid_eval, preds),
                "macro_f1": f1_score(y_valid_eval, preds, average="macro"),
            }
        )

    return {
        "fold_scores": fold_scores,
        "mean_accuracy": sum(x["accuracy"] for x in fold_scores) / len(fold_scores),
        "mean_macro_f1": sum(x["macro_f1"] for x in fold_scores) / len(fold_scores),
    }


def evaluate_holdout_model(
    model: Pipeline,
    use_sample_weight: bool,
    encode_target: bool,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    y_train_fit = y_train
    fit_kwargs = {}
    if encode_target:
        class_names = sorted(y_train.unique().tolist())
        label_to_int = {label: idx for idx, label in enumerate(class_names)}
        int_to_label = {idx: label for label, idx in label_to_int.items()}
        y_train_fit = y_train.map(label_to_int)
    if use_sample_weight:
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", y_train_fit)

    model.fit(X_train, y_train_fit, **fit_kwargs)
    preds = model.predict(X_test)
    if encode_target:
        preds = pd.Series(preds).map(int_to_label).to_numpy()

    labels = sorted(y_test.unique().tolist())
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=labels).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "predictions": preds.tolist(),
    }


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    X, y, groups = build_feature_table(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    preprocessor = make_preprocessor(X_train)
    model_specs = build_models(preprocessor)

    results = []
    predictions_df = df.iloc[test_idx][[GROUP_COLUMN, TARGET_COLUMN]].copy()

    for name, model, use_sample_weight in model_specs:
        encode_target = name in {"xgboost", "lightgbm"}
        cv_result = cross_validate_model(
            model,
            use_sample_weight,
            encode_target,
            X_train,
            y_train,
            train_groups,
        )
        holdout_result = evaluate_holdout_model(
            model,
            use_sample_weight,
            encode_target,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        predictions_df[f"{name}_prediction"] = holdout_result["predictions"]
        results.append(
            {
                "model": name,
                "cross_validation": cv_result,
                "holdout_accuracy": holdout_result["accuracy"],
                "holdout_macro_f1": holdout_result["macro_f1"],
                "labels": holdout_result["labels"],
                "confusion_matrix": holdout_result["confusion_matrix"],
                "classification_report": holdout_result["classification_report"],
            }
        )

    best_accuracy_model = max(
        results,
        key=lambda item: (
            item["holdout_accuracy"],
            item["cross_validation"]["mean_accuracy"],
        ),
    )["model"]
    best_macro_f1_model = max(
        results,
        key=lambda item: (
            item["cross_validation"]["mean_macro_f1"],
            item["holdout_macro_f1"],
        ),
    )["model"]

    results.sort(
        key=lambda item: (
            item["holdout_accuracy"],
            item["cross_validation"]["mean_accuracy"],
            item["holdout_macro_f1"],
        ),
        reverse=True,
    )

    summary = {
        "dataset_path": str(DATA_PATH),
        "total_rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_subjects": int(df.iloc[train_idx][GROUP_COLUMN].nunique()),
        "test_subjects": int(df.iloc[test_idx][GROUP_COLUMN].nunique()),
        "class_distribution": y.value_counts().to_dict(),
        "feature_count": int(X.shape[1]),
        "features": X.columns.tolist(),
        "models": results,
        "recommended_model_for_accuracy": best_accuracy_model,
        "recommended_model_for_macro_f1": best_macro_f1_model,
    }

    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")
    print(f"Saved test predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
