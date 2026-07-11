"""
ml_engine.py — Local Machine Learning Engine (Enhanced).
Trains, evaluates, and compares models using scikit-learn.
Now includes: cross-validation, confusion matrix, ROC curves,
gradient boosting, and permutation importance.
"""
from __future__ import annotations

import pickle
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_ml_data(
    df: pd.DataFrame, target_col: str
) -> Tuple[np.ndarray, np.ndarray, List[str], bool]:
    """
    Prepares a DataFrame for machine learning by encoding objects and handling NaNs.
    Returns: (X, y, feature_names, is_classification)
    """
    df = df.copy()

    # Drop non-predictive features (zero-variance columns)
    for col in df.columns:
        if df[col].nunique() <= 1:
            df = df.drop(columns=[col])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    # Drop target nulls
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Determine task type
    is_classification = False
    if y_raw.dtype == object or y_raw.dtype == bool or y_raw.nunique() < 15:
        is_classification = True

    # Encode target if classification
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_labels = le.classes_.tolist()
    else:
        y = y_raw.values.astype(float)
        class_labels = []

    # Encode features
    feature_names = []
    X_parts = []
    for col in X_raw.columns:
        col_data = X_raw[col]
        # Impute missing values
        if col_data.isnull().any():
            if pd.api.types.is_numeric_dtype(col_data):
                col_data = col_data.fillna(col_data.median())
            else:
                col_data = col_data.fillna(
                    col_data.mode().iloc[0] if not col_data.mode().empty else "Missing"
                )

        if pd.api.types.is_numeric_dtype(col_data):
            X_parts.append(col_data.values.reshape(-1, 1).astype(float))
            feature_names.append(col)
        else:
            # One-hot encode string columns
            dummies = pd.get_dummies(col_data, prefix=col, drop_first=True)
            X_parts.append(dummies.values.astype(float))
            feature_names.extend(dummies.columns.tolist())

    if not X_parts:
        raise ValueError("No valid features remaining for training after preprocessing.")

    X = np.hstack(X_parts)
    return X, y, feature_names, is_classification, class_labels


def _compute_permutation_importance(
    model, X_test: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], is_classification: bool, n_repeats: int = 5
) -> List[Dict]:
    """
    Compute permutation importance — model-agnostic, works for any estimator.
    Shuffles each feature and measures accuracy drop.
    """
    if is_classification:
        baseline = accuracy_score(y_test, model.predict(X_test))
    else:
        baseline = r2_score(y_test, model.predict(X_test))

    importances = []
    for i, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            if is_classification:
                score = accuracy_score(y_test, model.predict(X_perm))
            else:
                score = r2_score(y_test, model.predict(X_perm))
            scores.append(baseline - score)
        importances.append({
            "feature": name,
            "importance": float(np.mean(scores)),
            "std": float(np.std(scores)),
        })

    return sorted(importances, key=lambda x: x["importance"], reverse=True)[:20]


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    run_cv: bool = True,
    cv_folds: int = 5,
    compute_perm_importance: bool = False,
) -> Dict[str, Any]:
    """
    Trains multiple model architectures and compares their performance.
    Enhanced with cross-validation, confusion matrix, ROC curves, and gradient boosting.
    """
    try:
        X, y, feature_names, is_classification, class_labels = prepare_ml_data(df, target_col)
    except Exception as e:
        return {"error": f"Data preparation failed: {str(e)}"}

    if len(X) < 10:
        return {"error": "Need at least 10 rows to train a model."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if is_classification and len(np.unique(y)) > 1 else None,
    )

    results = {
        "is_classification": is_classification,
        "feature_names": feature_names,
        "class_labels": class_labels,
        "models": {},
        "best_model_name": None,
        "best_score": -1 if is_classification else float("-inf"),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    if is_classification:
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        }
        cv_scoring = "f1_weighted"
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            "Linear Regression": LinearRegression(),
        }
        cv_scoring = "r2"

    is_binary = is_classification and len(np.unique(y)) == 2

    for name, model in models.items():
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            fit_time = time.time() - start_time
            preds = model.predict(X_test)

            metrics = {"fit_time": fit_time}

            if is_classification:
                metrics["Accuracy"] = float(accuracy_score(y_test, preds))
                metrics["F1-Score"] = float(f1_score(y_test, preds, average="weighted", zero_division=0))
                metrics["Precision"] = float(precision_score(y_test, preds, average="weighted", zero_division=0))
                metrics["Recall"] = float(recall_score(y_test, preds, average="weighted", zero_division=0))

                # Confusion matrix
                cm = confusion_matrix(y_test, preds)
                metrics["confusion_matrix"] = cm.tolist()

                # ROC curve (binary only)
                if is_binary and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, proba)
                        roc_auc = auc(fpr, tpr)
                        metrics["roc"] = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "auc": float(roc_auc),
                        }
                    except Exception:
                        pass

                score = metrics["F1-Score"]
                if score > results["best_score"]:
                    results["best_score"] = score
                    results["best_model_name"] = name
            else:
                metrics["R2-Score"] = float(r2_score(y_test, preds))
                metrics["MAE"] = float(mean_absolute_error(y_test, preds))
                metrics["MSE"] = float(mean_squared_error(y_test, preds))
                metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))

                score = metrics["R2-Score"]
                if score > results["best_score"] or results["best_model_name"] is None:
                    results["best_score"] = score
                    results["best_model_name"] = name

            # Cross-validation
            if run_cv and len(X) >= cv_folds * 2:
                try:
                    cv_scores = cross_val_score(
                        model, X, y, cv=cv_folds, scoring=cv_scoring, n_jobs=-1
                    )
                    metrics["cv_mean"] = float(np.mean(cv_scores))
                    metrics["cv_std"] = float(np.std(cv_scores))
                    metrics["cv_scores"] = cv_scores.tolist()
                except Exception:
                    pass

            # Feature importance (tree-based)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                fi = [
                    {"feature": feature_names[i], "importance": float(importances[i])}
                    for i in indices[:20]
                ]
                metrics["feature_importances"] = fi

            elif name in ("Linear Regression", "Logistic Regression"):
                try:
                    coefs = model.coef_
                    if len(coefs.shape) > 1:
                        coefs = np.mean(np.abs(coefs), axis=0)
                    else:
                        coefs = np.abs(coefs.flatten())
                    if coefs.sum() > 0:
                        coefs = coefs / coefs.sum()
                    indices = np.argsort(coefs)[::-1]
                    fi = [
                        {"feature": feature_names[i], "importance": float(coefs[i])}
                        for i in indices[:20]
                        if i < len(feature_names)
                    ]
                    metrics["feature_importances"] = fi
                except Exception:
                    pass

            # Permutation importance (optional, slower)
            if compute_perm_importance and len(X_test) >= 20:
                try:
                    perm_fi = _compute_permutation_importance(
                        model, X_test, y_test, feature_names, is_classification, n_repeats=3
                    )
                    metrics["permutation_importances"] = perm_fi
                except Exception:
                    pass

            results["models"][name] = {
                "metrics": metrics,
                "model_bytes": pickle.dumps(model),
                "predictions": preds.tolist(),
                "actuals": y_test.tolist(),
                "model_obj": model,  # Keep in-memory for PDP
            }

        except Exception as e:
            results["models"][name] = {"error": str(e)}

    return results
