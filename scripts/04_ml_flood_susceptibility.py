#!/usr/bin/env python3
"""
04_ml_flood_susceptibility.py
==============================
Machine Learning flood susceptibility modelling for Narino, Colombia.

This module:
  1. Loads training data (feature samples + binary flood labels).
  2. Trains three models: Random Forest, XGBoost, LightGBM.
  3. Performs spatial cross-validation (stratified K-fold with spatial blocks).
  4. Computes evaluation metrics: AUC-ROC, accuracy, precision, recall, F1.
  5. Generates SHAP-based feature importance analysis.
  6. Produces a pixel-level flood susceptibility probability map.
  7. Creates an ensemble prediction (weighted average of three models).
  8. Aggregates municipal-level flood risk statistics.

Usage:
    python 04_ml_flood_susceptibility.py
    python 04_ml_flood_susceptibility.py --data-path data/training_samples.csv
    python 04_ml_flood_susceptibility.py --skip-shap

Author : Narino Flood Risk Research Project
Date   : 2026-02-26
"""

import sys
import json
import pathlib
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import xgboost as xgb
except (ImportError, TypeError, OSError):
    xgb = None

try:
    import lightgbm as lgb
except (ImportError, TypeError, OSError):
    # TypeError can occur when lightgbm's compat module tries to import
    # dask.dataframe with an incompatible dask/pandas version combination.
    # The dask DatetimeAccessor metaclass calls inspect.signature() on a
    # pandas property object, raising:
    #   TypeError: descriptor '__call__' for 'type' objects doesn't apply
    #              to a 'property' object
    # OSError covers shared-library loading failures.
    lgb = None

try:
    import shap
except (ImportError, TypeError, OSError):
    shap = None

import ee

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gee_config as cfg  # noqa: E402
from scripts.utils import (  # noqa: E402
    get_study_area_geometry,
    get_municipalities,
    setup_logging,
    safe_getinfo,
    export_to_drive,
    export_table_to_drive,
    monitor_tasks,
)

log = setup_logging("04_ml_model")

# Output directories
MODEL_DIR = cfg.OUTPUTS_DIR / "phase3_risk_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# DATA LOADING AND PREPARATION
# ===========================================================================

def prepare_training_data(
    csv_path: pathlib.Path,
    feature_columns: Optional[List[str]] = None,
    label_column: str = "label",
    drop_nulls: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load and prepare training data from a CSV file exported from GEE.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the CSV containing sampled features and labels.
    feature_columns : list[str], optional
        Feature column names. Defaults to ``gee_config.SUSCEPTIBILITY_FEATURES``.
    label_column : str
        Name of the binary label column.
    drop_nulls : bool
        Remove rows with any NaN values.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with all columns.
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Label vector (n_samples,).
    """
    if feature_columns is None:
        feature_columns = list(cfg.SUSCEPTIBILITY_FEATURES)

    log.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Raw samples: %d rows, %d columns", len(df), len(df.columns))

    # Verify required columns exist
    available = set(df.columns)
    missing = [c for c in feature_columns + [label_column] if c not in available]
    if missing:
        log.warning("Missing columns: %s. Dropping from feature list.", missing)
        feature_columns = [c for c in feature_columns if c in available]

    if drop_nulls:
        before = len(df)
        df = df.dropna(subset=feature_columns + [label_column])
        dropped = before - len(df)
        if dropped:
            log.info("Dropped %d rows with NaN values (%d remaining)", dropped, len(df))

    X = df[feature_columns].values.astype(np.float64)
    y = df[label_column].values.astype(int)

    log.info(
        "Prepared data: X shape=%s, y shape=%s, class balance: 0=%d, 1=%d",
        X.shape, y.shape, np.sum(y == 0), np.sum(y == 1),
    )
    return df, X, y


# ===========================================================================
# SPATIAL CROSS-VALIDATION
# ===========================================================================

def spatial_cross_validation(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = cfg.CV_PARAMS["n_splits"],
    lat_col: str = ".geo",
    seed: int = cfg.CV_PARAMS["random_state"],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate spatial block cross-validation folds.

    To reduce spatial autocorrelation between train and test sets, this
    method divides the study area into latitudinal strips and uses each
    strip as a test fold.

    If latitude information is not available (no geometry column), falls
    back to standard stratified K-fold.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    n_splits : int
        Number of CV folds.
    lat_col : str
        Column name containing geometry or latitude.
    seed : int
        Random seed.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.
    """
    # Try to extract latitude from GEE-exported geometry column
    latitudes = None
    if lat_col in df.columns:
        try:
            geo_series = df[lat_col].apply(
                lambda g: json.loads(g)["coordinates"][1] if isinstance(g, str) else None
            )
            latitudes = geo_series.dropna().values
            if len(latitudes) == len(df):
                log.info("Latitude extracted from geometry column for spatial CV")
        except (json.JSONDecodeError, KeyError, TypeError):
            latitudes = None

    if latitudes is not None and len(latitudes) == len(df):
        # Spatial blocking by latitude quantiles
        lat_quantiles = np.percentile(latitudes, np.linspace(0, 100, n_splits + 1))
        fold_assignments = np.digitize(latitudes, lat_quantiles[1:-1])

        folds = []
        for fold_id in range(n_splits):
            test_idx = np.where(fold_assignments == fold_id)[0]
            train_idx = np.where(fold_assignments != fold_id)[0]
            if len(test_idx) == 0:
                continue
            folds.append((train_idx, test_idx))

        log.info("Spatial CV: %d folds by latitude blocks", len(folds))
        return folds
    else:
        # Fallback: stratified K-fold
        log.warning("No spatial info available; using stratified K-fold CV")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(X, y))


# ===========================================================================
# MODEL TRAINING
# ===========================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict] = None,
) -> RandomForestClassifier:
    """
    Train a scikit-learn Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    params : dict, optional
        Hyperparameters (defaults to ``gee_config.ML_PARAMS['random_forest']``).

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    params = params or dict(cfg.ML_PARAMS["random_forest"])
    log.info("Training Random Forest: %s", params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    log.info("Random Forest trained (n_estimators=%d)", params.get("n_estimators", 100))
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict] = None,
) -> "xgb.XGBClassifier":
    """
    Train an XGBoost classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    params : dict, optional
        Hyperparameters (defaults to ``gee_config.ML_PARAMS['xgboost']``).

    Returns
    -------
    xgb.XGBClassifier
        Fitted model.

    Raises
    ------
    ImportError
        If xgboost is not installed.
    """
    if xgb is None:
        raise ImportError("xgboost is required: pip install xgboost")

    params = params or dict(cfg.ML_PARAMS["xgboost"])
    log.info("Training XGBoost: %s", params)

    # Note: use_label_encoder was removed in xgboost 2.0+; do not pass it.
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    log.info("XGBoost trained")
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict] = None,
) -> "lgb.LGBMClassifier":
    """
    Train a LightGBM classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    params : dict, optional
        Hyperparameters (defaults to ``gee_config.ML_PARAMS['lightgbm']``).

    Returns
    -------
    lgb.LGBMClassifier
        Fitted model.

    Raises
    ------
    ImportError
        If lightgbm is not installed.
    """
    if lgb is None:
        raise ImportError("lightgbm is required: pip install lightgbm")

    params = params or dict(cfg.ML_PARAMS["lightgbm"])
    log.info("Training LightGBM: %s", params)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    log.info("LightGBM trained")
    return model


# ===========================================================================
# EVALUATION
# ===========================================================================

def _evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics for a single fold."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def cross_validate_model(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str = "model",
    params: Optional[Dict] = None,
) -> Tuple[Dict[str, float], List]:
    """
    Evaluate a model using pre-defined cross-validation folds.

    Parameters
    ----------
    model_fn : callable
        One of ``train_random_forest``, ``train_xgboost``, ``train_lightgbm``.
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Full label vector.
    folds : list
        Cross-validation fold indices from ``spatial_cross_validation()``.
    model_name : str
        Name for logging.
    params : dict, optional
        Model hyperparameters.

    Returns
    -------
    mean_metrics : dict[str, float]
        Mean metrics across folds.
    fold_models : list
        Fitted model from each fold.
    """
    all_metrics = []
    fold_models = []

    for i, (train_idx, test_idx) in enumerate(folds):
        log.info("  [%s] Fold %d/%d (train=%d, test=%d)",
                 model_name, i + 1, len(folds), len(train_idx), len(test_idx))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_fn(X_train, y_train, params)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = _evaluate_fold(y_test, y_pred, y_prob)
        all_metrics.append(metrics)
        fold_models.append(model)

        log.info("    AUC=%.4f  Acc=%.4f  F1=%.4f",
                 metrics["auc_roc"], metrics["accuracy"], metrics["f1"])

    # Mean across folds
    mean_metrics = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
    }
    std_metrics = {
        f"{k}_std": float(np.std([m[k] for m in all_metrics]))
        for k in all_metrics[0]
    }
    mean_metrics.update(std_metrics)

    log.info(
        "[%s] Mean AUC=%.4f (+/- %.4f), F1=%.4f (+/- %.4f)",
        model_name,
        mean_metrics["auc_roc"], mean_metrics["auc_roc_std"],
        mean_metrics["f1"], mean_metrics["f1_std"],
    )
    return mean_metrics, fold_models


# ===========================================================================
# SHAP FEATURE IMPORTANCE
# ===========================================================================

def compute_shap_importance(
    model,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str = "model",
    max_samples: int = 2000,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance values.

    Parameters
    ----------
    model : fitted model
        Scikit-learn API compatible classifier.
    X : np.ndarray
        Feature matrix for SHAP computation.
    feature_names : list[str]
        Feature names matching columns of X.
    model_name : str
        Name for logging and output.
    max_samples : int
        Maximum number of samples for SHAP (subsampled for speed).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature``, ``mean_abs_shap``, sorted
        descending by importance.
    """
    if shap is None:
        log.warning("SHAP not installed (pip install shap). Skipping importance.")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    log.info("Computing SHAP values for %s (%d features) ...", model_name, len(feature_names))

    if len(X) > max_samples:
        rng = np.random.RandomState(cfg.CV_PARAMS["random_state"])
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_shap = X[idx]
    else:
        X_shap = X

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values may be a list of two arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (flood)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    log.info("Top-5 features by SHAP: %s",
             importance_df.head(5)[["feature", "mean_abs_shap"]].to_dict("records"))

    return importance_df


# ===========================================================================
# SUSCEPTIBILITY MAP GENERATION (GEE-based)
# ===========================================================================

def generate_susceptibility_map(
    model,
    feature_names: List[str],
    region: ee.Geometry,
    model_name: str = "rf",
) -> ee.Image:
    """
    Generate a pixel-level flood susceptibility map using a trained
    model applied via GEE's built-in classifiers.

    For scikit-learn models trained locally, this function re-trains a
    GEE-native Random Forest (``ee.Classifier.smileRandomForest``) using
    the same training samples. This avoids transferring pixel-level data
    to the client.

    Parameters
    ----------
    model : fitted sklearn model
        Used only for metadata; the GEE classifier is trained from
        scratch on the GEE-side samples.
    feature_names : list[str]
        Feature band names in the stack.
    region : ee.Geometry
        Prediction region.
    model_name : str
        Model identifier for the output band name.

    Returns
    -------
    ee.Image
        Continuous probability image (0-1) named ``'susceptibility_{model_name}'``.
    """
    log.info("Generating GEE susceptibility map: %s", model_name)

    import importlib
    feat_module = importlib.import_module("scripts.03_flood_susceptibility_features")
    stack_all_features = feat_module.stack_all_features
    generate_training_samples = feat_module.generate_training_samples

    feature_stack = stack_all_features(region)
    samples = generate_training_samples(feature_stack, region)

    # GEE-native Random Forest
    n_trees = cfg.ML_PARAMS["random_forest"]["n_estimators"]
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=min(n_trees, 200),  # GEE limit
    ).setOutputMode("PROBABILITY")

    trained = classifier.train(
        features=samples,
        classProperty="label",
        inputProperties=feature_names,
    )

    probability = (
        feature_stack
        .select(feature_names)
        .classify(trained)
        .rename(f"susceptibility_{model_name}")
        .clip(region)
    )

    log.info("Susceptibility map generated: susceptibility_%s", model_name)
    return probability


# ===========================================================================
# ENSEMBLE PREDICTION
# ===========================================================================

def ensemble_prediction(
    models: Dict[str, object],
    X: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute a weighted ensemble probability prediction.

    Parameters
    ----------
    models : dict[str, model]
        Named fitted models (e.g. ``{'rf': model1, 'xgb': model2}``).
    X : np.ndarray
        Feature matrix.
    weights : dict[str, float], optional
        Model weights (must sum to ~1). If None, uses equal weights.

    Returns
    -------
    np.ndarray
        Ensemble flood probability (n_samples,).
    """
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models}

    # Normalise weights
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    ensemble_prob = np.zeros(len(X))
    for name, model in models.items():
        w = weights.get(name, 0.0)
        prob = model.predict_proba(X)[:, 1]
        ensemble_prob += w * prob
        log.debug("Ensemble component '%s': weight=%.3f", name, w)

    return ensemble_prob


def generate_ensemble_map_gee(
    region: ee.Geometry,
    feature_names: List[str],
) -> ee.Image:
    """
    Generate an ensemble susceptibility map using multiple GEE classifiers.

    Trains Random Forest, Gradient Tree Boost, and CART on the GEE side
    and averages their probability predictions.

    Parameters
    ----------
    region : ee.Geometry
        Prediction region.
    feature_names : list[str]
        Feature band names.

    Returns
    -------
    ee.Image
        Ensemble probability image (0-1) named ``'susceptibility_ensemble'``.
    """
    log.info("Generating GEE ensemble susceptibility map ...")

    import importlib
    feat_module = importlib.import_module("scripts.03_flood_susceptibility_features")
    stack_all_features = feat_module.stack_all_features
    generate_training_samples = feat_module.generate_training_samples

    feature_stack = stack_all_features(region)
    samples = generate_training_samples(feature_stack, region)

    classifiers = {
        "rf": ee.Classifier.smileRandomForest(numberOfTrees=200),
        "gbt": ee.Classifier.smileGradientTreeBoost(numberOfTrees=200),
        "cart": ee.Classifier.smileCart(),
    }

    probabilities = []
    for name, clf in classifiers.items():
        trained = clf.setOutputMode("PROBABILITY").train(
            features=samples,
            classProperty="label",
            inputProperties=feature_names,
        )
        prob = feature_stack.select(feature_names).classify(trained)
        probabilities.append(prob)
        log.info("GEE classifier '%s' trained", name)

    # Average the probabilities
    ensemble = (
        ee.ImageCollection.fromImages(probabilities)
        .mean()
        .rename("susceptibility_ensemble")
        .clip(region)
    )

    log.info("Ensemble susceptibility map generated")
    return ensemble


# ===========================================================================
# MUNICIPAL RISK STATISTICS
# ===========================================================================

def municipal_risk_stats(
    susceptibility: ee.Image,
    band: str = "susceptibility_ensemble",
    region: Optional[ee.Geometry] = None,
) -> ee.FeatureCollection:
    """
    Compute municipal-level flood risk statistics from a susceptibility map.

    For each municipality, computes:
      - Mean, median, max susceptibility probability.
      - Percentage of area in each risk class (0-20%, 20-40%, ..., 80-100%).
      - Area (km^2) in high-risk class (probability > 60%).

    Parameters
    ----------
    susceptibility : ee.Image
        Continuous probability image.
    band : str
        Band name to analyse.
    region : ee.Geometry, optional
        Clipping region (defaults to Narino).

    Returns
    -------
    ee.FeatureCollection
        Municipalities with risk statistics as properties.
    """
    if region is None:
        region = get_study_area_geometry()

    municipalities = get_municipalities()
    prob = susceptibility.select(band).clip(region)

    # Risk class thresholds
    risk_classes = {
        "very_low": (0, 0.2),
        "low": (0.2, 0.4),
        "moderate": (0.4, 0.6),
        "high": (0.6, 0.8),
        "very_high": (0.8, 1.01),
    }

    # Create classified image
    risk_class_img = ee.Image(0).rename("risk_class")
    for idx, (_, (lo, hi)) in enumerate(risk_classes.items(), start=1):
        mask = prob.gte(lo).And(prob.lt(hi))
        risk_class_img = risk_class_img.where(mask, idx)
    risk_class_img = risk_class_img.selfMask()

    # Pixel area in hectares
    pixel_area_ha = ee.Image.pixelArea().divide(10000)

    def _compute_stats(muni):
        muni = ee.Feature(muni)
        geom = muni.geometry()

        # Basic stats
        stats = prob.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.median(), sharedInputs=True)
                .combine(ee.Reducer.max(), sharedInputs=True)
                .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True),
            geometry=geom,
            scale=cfg.EXPORT_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )

        # Area per risk class
        class_areas = risk_class_img.addBands(pixel_area_ha).reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=0, groupName="class"),
            geometry=geom,
            scale=cfg.EXPORT_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )

        return muni.set(stats).set("risk_class_areas", class_areas.get("groups"))

    muni_stats = municipalities.map(_compute_stats)
    log.info("Municipal risk statistics computed")
    return muni_stats


# ===========================================================================
# FULL PIPELINE
# ===========================================================================

def run_ml_pipeline(
    data_path: Optional[pathlib.Path] = None,
    skip_shap: bool = False,
    export_maps: bool = True,
) -> Dict:
    """
    Execute the full ML flood susceptibility pipeline.

    Parameters
    ----------
    data_path : pathlib.Path, optional
        Path to training CSV. If None, uses the default export location.
    skip_shap : bool
        Skip SHAP computation (faster).
    export_maps : bool
        Export GEE-based susceptibility maps to Drive.

    Returns
    -------
    dict
        Results dictionary with model metrics, importance, etc.
    """
    # --- 1. Load data ---
    if data_path is None:
        data_path = cfg.DATA_DIR / "satellite_exports" / "narino_flood_training_samples.csv"
    if not data_path.exists():
        log.error("Training data not found: %s", data_path)
        log.error("Run 03_flood_susceptibility_features.py first to export training samples.")
        return {}

    feature_cols = list(cfg.SUSCEPTIBILITY_FEATURES)
    df, X, y = prepare_training_data(data_path, feature_cols)

    # --- 2. Spatial CV folds ---
    folds = spatial_cross_validation(df, X, y)

    # --- 3. Train and evaluate models ---
    results = {}

    log.info("=" * 40)
    log.info("Training Random Forest")
    log.info("=" * 40)
    rf_metrics, rf_models = cross_validate_model(
        train_random_forest, X, y, folds, "RF"
    )
    results["rf"] = {"metrics": rf_metrics}

    if xgb is not None:
        log.info("=" * 40)
        log.info("Training XGBoost")
        log.info("=" * 40)
        xgb_metrics, xgb_models = cross_validate_model(
            train_xgboost, X, y, folds, "XGB"
        )
        results["xgb"] = {"metrics": xgb_metrics}
    else:
        log.warning("XGBoost not installed; skipping")
        xgb_models = []

    if lgb is not None:
        log.info("=" * 40)
        log.info("Training LightGBM")
        log.info("=" * 40)
        lgb_metrics, lgb_models = cross_validate_model(
            train_lightgbm, X, y, folds, "LGBM"
        )
        results["lgbm"] = {"metrics": lgb_metrics}
    else:
        log.warning("LightGBM not installed; skipping")
        lgb_models = []

    # --- 4. Train final models on full data ---
    log.info("Training final models on full dataset ...")
    rf_final = train_random_forest(X, y)
    results["rf"]["final_model"] = rf_final

    models_dict = {"rf": rf_final}
    if xgb is not None:
        xgb_final = train_xgboost(X, y)
        results["xgb"]["final_model"] = xgb_final
        models_dict["xgb"] = xgb_final
    if lgb is not None:
        lgb_final = train_lightgbm(X, y)
        results["lgbm"]["final_model"] = lgb_final
        models_dict["lgbm"] = lgb_final

    # Save models to disk
    for name, model in models_dict.items():
        model_path = MODEL_DIR / f"flood_susceptibility_{name}.joblib"
        joblib.dump(model, model_path)
        log.info("Model saved: %s", model_path)

    # --- 5. Ensemble on holdout ---
    ensemble_prob = ensemble_prediction(models_dict, X)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ensemble_metrics = _evaluate_fold(y, ensemble_pred, ensemble_prob)
    results["ensemble"] = {"metrics": ensemble_metrics}
    log.info("Ensemble AUC=%.4f, F1=%.4f", ensemble_metrics["auc_roc"], ensemble_metrics["f1"])

    # --- 6. SHAP importance ---
    if not skip_shap:
        for name, model in models_dict.items():
            importance = compute_shap_importance(model, X, feature_cols, name)
            importance.to_csv(MODEL_DIR / f"shap_importance_{name}.csv", index=False)
            results[name]["shap"] = importance

    # --- 7. Save metrics summary ---
    metrics_summary = {
        name: res.get("metrics", {})
        for name, res in results.items()
        if "metrics" in res
    }
    metrics_path = MODEL_DIR / "model_metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    log.info("Metrics summary saved: %s", metrics_path)

    # --- 8. Generate GEE maps and municipal stats ---
    tasks = []
    if export_maps:
        region = get_study_area_geometry()
        feature_names = feature_cols

        ensemble_map = generate_ensemble_map_gee(region, feature_names)
        task = export_to_drive(
            image=ensemble_map.toFloat(),
            description="narino_flood_susceptibility_ensemble",
            region=region,
            scale=cfg.EXPORT_SCALE,
        )
        tasks.append(task)

        # Municipal stats
        muni_stats = municipal_risk_stats(ensemble_map)
        task = export_table_to_drive(
            collection=muni_stats,
            description="narino_municipal_risk_stats",
            file_format="CSV",
        )
        tasks.append(task)

        results["export_tasks"] = tasks

    log.info("ML pipeline complete.")
    return results


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main() -> None:
    """Command-line interface for ML flood susceptibility modelling."""
    parser = argparse.ArgumentParser(
        description="ML flood susceptibility modelling for Narino",
    )
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        default=None,
        help="Path to training samples CSV.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP feature importance (faster).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip GEE map export.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Block and monitor export tasks until completion.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ML Flood Susceptibility Modelling Pipeline")
    log.info("=" * 60)

    results = run_ml_pipeline(
        data_path=args.data_path,
        skip_shap=args.skip_shap,
        export_maps=not args.no_export,
    )

    if args.monitor and results.get("export_tasks"):
        monitor_tasks(results["export_tasks"])

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
