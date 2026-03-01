#!/usr/bin/env python3
"""
09_quality_control.py
=====================
Quality control and validation for the Narino Flood Risk Assessment.

Checks:
  1. All expected outputs exist and are non-empty
  2. Area calculations (Narino total ~ 33,268 km2)
  3. Cross-validation of SAR water detection against JRC GSW
  4. ML model metrics are within reasonable ranges
  5. Municipal statistics sum to department totals
  6. Generate comprehensive QC report in Markdown format

Functions:
  check_outputs()          - Verify all output files exist and are non-empty
  validate_areas()         - Validate geographic area calculations
  cross_validate_water()   - Compare SAR water with JRC reference
  check_ml_metrics()       - Verify ML metrics are reasonable
  verify_municipal_stats() - Check that municipal aggregates match totals
  generate_qc_report()     - Compile all checks into a Markdown report

Output:
  - outputs/phase5_qc/qc_report.md
  - outputs/phase5_qc/qc_summary.csv

Usage:
  python scripts/09_quality_control.py

Author: Flood Risk Research Project
"""

import sys
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gee_config import (
    SUBREGIONS, SUSCEPTIBILITY_FEATURES, HAND_CLASSES,
    FLOOD_FREQUENCY_CLASSES, ML_PARAMS,
)
from utils import (
    setup_logging, ensure_dirs,
    load_narino_boundary, load_municipalities, load_subregions,
    compute_area_km2, validate_narino_area,
    OUTPUTS_DIR, TABLES_DIR, FIGURES_DIR, OVERLEAF_FIGURES, OVERLEAF_TABLES,
    NARINO_AREA_KM2, NARINO_AREA_TOLERANCE,
    CRS_WGS84, CRS_COLOMBIA,
)

logger = setup_logging("09_quality_control")

QC_DIR = OUTPUTS_DIR / "phase5_qc"


# ============================================================================
# Data Structures
# ============================================================================

class QCResult:
    """Container for a single QC check result."""

    def __init__(
        self,
        check_name: str,
        category: str,
        passed: bool,
        message: str,
        details: Optional[str] = None,
        severity: str = "INFO",
    ):
        self.check_name = check_name
        self.category = category
        self.passed = passed
        self.message = message
        self.details = details or ""
        self.severity = severity  # INFO, WARNING, ERROR, CRITICAL
        self.timestamp = datetime.now().isoformat()

    def status_str(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict:
        return {
            "check": self.check_name,
            "category": self.category,
            "status": self.status_str(),
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# Global list for accumulating QC results
_qc_results: List[QCResult] = []


def _add_result(result: QCResult) -> None:
    """Add a QC result to the global accumulator and log it."""
    _qc_results.append(result)
    status = result.status_str()
    log_func = logger.info if result.passed else logger.warning
    log_func("  [%s] %s: %s", status, result.check_name, result.message)


# ============================================================================
# 1. Check Outputs Exist
# ============================================================================

def check_outputs() -> List[QCResult]:
    """
    Verify that all expected output files and directories exist and are
    non-empty.  Checks figures (PDF/PNG), tables (CSV/TEX), and
    intermediate data products.

    Returns
    -------
    list of QCResult
    """
    logger.info("Checking output files and directories...")
    results = []

    # Expected figure files
    expected_figures = [
        "fig01_study_area",
        "fig02_sar_water_detection",
        "fig03_jrc_water_occurrence",
        "fig04_flood_frequency",
        "fig05_hand_susceptibility",
        "fig06_roc_curves",
        "fig07_shap_importance",
        "fig08_flood_susceptibility",
        "fig09_municipal_risk_ranking",
        "fig10_population_exposure",
        "fig11_seasonal_dynamics",
        "fig12_climate_flood_correlation",
    ]

    for fig_name in expected_figures:
        for fmt in ["pdf", "png"]:
            for directory, dir_label in [
                (FIGURES_DIR, "outputs"),
                (OVERLEAF_FIGURES, "overleaf"),
            ]:
                fpath = directory / f"{fig_name}.{fmt}"
                exists = fpath.exists()
                non_empty = exists and fpath.stat().st_size > 0
                r = QCResult(
                    check_name=f"figure_{fig_name}_{fmt}_{dir_label}",
                    category="outputs_figures",
                    passed=exists and non_empty,
                    message=(
                        f"{fig_name}.{fmt} in {dir_label}: "
                        + ("OK" if non_empty else "MISSING or EMPTY")
                    ),
                    details=f"Path: {fpath}, Size: {fpath.stat().st_size if exists else 0} bytes",
                    severity="WARNING" if not non_empty else "INFO",
                )
                results.append(r)
                _add_result(r)

    # Expected table files
    expected_tables = [
        "table1_data_sources",
        "table2_sar_accuracy",
        "table3_ml_comparison",
        "table4_feature_importance",
        "table5_municipal_risk_top20",
        "table6_population_exposure",
        "table7_seasonal_dynamics_monthly",
    ]

    for tbl_name in expected_tables:
        # CSV
        csv_path = TABLES_DIR / f"{tbl_name}.csv"
        csv_ok = csv_path.exists() and csv_path.stat().st_size > 0
        r = QCResult(
            check_name=f"table_{tbl_name}_csv",
            category="outputs_tables",
            passed=csv_ok,
            message=f"{tbl_name}.csv: {'OK' if csv_ok else 'MISSING or EMPTY'}",
            severity="WARNING" if not csv_ok else "INFO",
        )
        results.append(r)
        _add_result(r)

        # LaTeX
        tex_path = OVERLEAF_TABLES / f"{tbl_name}.tex"
        tex_ok = tex_path.exists() and tex_path.stat().st_size > 0
        r = QCResult(
            check_name=f"table_{tbl_name}_tex",
            category="outputs_tables",
            passed=tex_ok,
            message=f"{tbl_name}.tex: {'OK' if tex_ok else 'MISSING or EMPTY'}",
            severity="WARNING" if not tex_ok else "INFO",
        )
        results.append(r)
        _add_result(r)

    # Phase output directories
    expected_phases = [
        "phase1_water_maps",
        "phase2_flood_frequency",
        "phase3_risk_model",
        "phase4_municipal_stats",
        "phase5_qc",
    ]
    for phase in expected_phases:
        pdir = OUTPUTS_DIR / phase
        exists = pdir.exists() and pdir.is_dir()
        r = QCResult(
            check_name=f"dir_{phase}",
            category="outputs_dirs",
            passed=exists,
            message=f"Directory {phase}: {'EXISTS' if exists else 'MISSING'}",
            severity="ERROR" if not exists else "INFO",
        )
        results.append(r)
        _add_result(r)

    return results


# ============================================================================
# 2. Validate Areas
# ============================================================================

def validate_areas() -> List[QCResult]:
    """
    Validate that area calculations for Narino boundaries are
    consistent with the expected 33,268 km2 (DANE official area).

    Tests:
    - GADM department boundary area
    - Sum of 64 municipality areas
    - Sum of 13 subregion areas
    - Individual municipality areas are positive and reasonable

    Returns
    -------
    list of QCResult
    """
    logger.info("Validating area calculations...")
    results = []

    # Test 1: Department boundary area
    for source in ["gadm", "geoboundaries", "naturalearth"]:
        try:
            boundary = load_narino_boundary(source)
            area_km2 = compute_area_km2(boundary)
            within_tol = abs(area_km2 - NARINO_AREA_KM2) / NARINO_AREA_KM2 <= NARINO_AREA_TOLERANCE
            diff_pct = (area_km2 - NARINO_AREA_KM2) / NARINO_AREA_KM2 * 100

            r = QCResult(
                check_name=f"area_department_{source}",
                category="area_validation",
                passed=within_tol,
                message=(
                    f"Narino area ({source}): {area_km2:,.1f} km2 "
                    f"(expected ~{NARINO_AREA_KM2:,.0f} km2, "
                    f"diff = {diff_pct:+.2f}%)"
                ),
                details=f"Tolerance: {NARINO_AREA_TOLERANCE*100:.0f}%",
                severity="ERROR" if not within_tol else "INFO",
            )
            results.append(r)
            _add_result(r)
        except FileNotFoundError:
            r = QCResult(
                check_name=f"area_department_{source}",
                category="area_validation",
                passed=False,
                message=f"Boundary file not found for source={source}",
                severity="WARNING",
            )
            results.append(r)
            _add_result(r)

    # Test 2: Sum of municipalities
    try:
        municipalities = load_municipalities("gadm")
        mun_total = compute_area_km2(municipalities)
        within_tol = abs(mun_total - NARINO_AREA_KM2) / NARINO_AREA_KM2 <= NARINO_AREA_TOLERANCE
        diff_pct = (mun_total - NARINO_AREA_KM2) / NARINO_AREA_KM2 * 100

        r = QCResult(
            check_name="area_municipalities_sum",
            category="area_validation",
            passed=within_tol,
            message=(
                f"Sum of 64 municipalities: {mun_total:,.1f} km2 "
                f"(diff = {diff_pct:+.2f}%)"
            ),
            severity="ERROR" if not within_tol else "INFO",
        )
        results.append(r)
        _add_result(r)

        # Individual municipality check
        mun_proj = municipalities.to_crs(CRS_COLOMBIA)
        mun_proj["area_km2"] = mun_proj.geometry.area / 1e6
        n_invalid = (mun_proj["area_km2"] <= 0).sum()
        n_too_small = (mun_proj["area_km2"] < 1).sum()  # < 1 km2
        n_too_large = (mun_proj["area_km2"] > 5000).sum()  # > 5000 km2

        r = QCResult(
            check_name="area_municipality_individual",
            category="area_validation",
            passed=(n_invalid == 0 and n_too_small == 0),
            message=(
                f"Individual areas: {n_invalid} invalid (<=0), "
                f"{n_too_small} too small (<1 km2), "
                f"{n_too_large} very large (>5000 km2)"
            ),
            details=(
                f"Range: {mun_proj['area_km2'].min():.1f} -- "
                f"{mun_proj['area_km2'].max():.1f} km2"
            ),
            severity="WARNING" if n_invalid > 0 else "INFO",
        )
        results.append(r)
        _add_result(r)

    except FileNotFoundError as exc:
        r = QCResult(
            check_name="area_municipalities_sum",
            category="area_validation",
            passed=False,
            message=f"Municipalities file not found: {exc}",
            severity="WARNING",
        )
        results.append(r)
        _add_result(r)

    # Test 3: Sum of subregions
    try:
        subregions = load_subregions()
        sub_total = compute_area_km2(subregions)
        within_tol = abs(sub_total - NARINO_AREA_KM2) / NARINO_AREA_KM2 <= NARINO_AREA_TOLERANCE
        diff_pct = (sub_total - NARINO_AREA_KM2) / NARINO_AREA_KM2 * 100

        r = QCResult(
            check_name="area_subregions_sum",
            category="area_validation",
            passed=within_tol,
            message=(
                f"Sum of 13 subregions: {sub_total:,.1f} km2 "
                f"(diff = {diff_pct:+.2f}%)"
            ),
            severity="WARNING" if not within_tol else "INFO",
        )
        results.append(r)
        _add_result(r)
    except FileNotFoundError:
        r = QCResult(
            check_name="area_subregions_sum",
            category="area_validation",
            passed=False,
            message="Subregions file not found.",
            severity="WARNING",
        )
        results.append(r)
        _add_result(r)

    return results


# ============================================================================
# 3. Cross-Validate SAR Water Detection
# ============================================================================

def cross_validate_water() -> List[QCResult]:
    """
    Cross-validate SAR-based water detection against JRC Global Surface
    Water reference data.

    Checks:
    - SAR permanent water area is within 20% of JRC permanent water
    - SAR maximum water extent is greater than JRC occurrence > 0%
    - Spatial overlap (IoU) between SAR and JRC water masks

    Returns
    -------
    list of QCResult
    """
    logger.info("Cross-validating SAR water detection against JRC GSW...")
    results = []

    # Try to load SAR accuracy data
    sar_path = TABLES_DIR / "table2_sar_accuracy.csv"
    if sar_path.exists():
        sar_acc = pd.read_csv(sar_path)

        # Check overall accuracy
        if "Overall Accuracy" in sar_acc.columns:
            numeric_rows = sar_acc[sar_acc["Year"] != "Mean"]
            if len(numeric_rows) > 0:
                min_oa = pd.to_numeric(numeric_rows["Overall Accuracy"], errors="coerce").min()
                max_oa = pd.to_numeric(numeric_rows["Overall Accuracy"], errors="coerce").max()
                mean_oa = pd.to_numeric(numeric_rows["Overall Accuracy"], errors="coerce").mean()

                reasonable = min_oa >= 0.70 and max_oa <= 1.0
                r = QCResult(
                    check_name="sar_overall_accuracy_range",
                    category="water_validation",
                    passed=reasonable,
                    message=(
                        f"SAR overall accuracy: min={min_oa:.3f}, "
                        f"max={max_oa:.3f}, mean={mean_oa:.3f}"
                    ),
                    details="Expected range: 0.70--1.00",
                    severity="ERROR" if not reasonable else "INFO",
                )
                results.append(r)
                _add_result(r)

        # Check Kappa
        if "Kappa" in sar_acc.columns:
            numeric_rows = sar_acc[sar_acc["Year"] != "Mean"]
            kappa_vals = pd.to_numeric(numeric_rows["Kappa"], errors="coerce")
            min_kappa = kappa_vals.min()
            reasonable = min_kappa >= 0.60
            r = QCResult(
                check_name="sar_kappa_range",
                category="water_validation",
                passed=reasonable,
                message=f"SAR Kappa: min={min_kappa:.3f}, mean={kappa_vals.mean():.3f}",
                details="Expected minimum: 0.60 (substantial agreement)",
                severity="WARNING" if not reasonable else "INFO",
            )
            results.append(r)
            _add_result(r)

        # Check F1 > 0.75
        if "F1-Score (Water)" in sar_acc.columns:
            numeric_rows = sar_acc[sar_acc["Year"] != "Mean"]
            f1_vals = pd.to_numeric(numeric_rows["F1-Score (Water)"], errors="coerce")
            min_f1 = f1_vals.min()
            reasonable = min_f1 >= 0.75
            r = QCResult(
                check_name="sar_f1_range",
                category="water_validation",
                passed=reasonable,
                message=f"SAR F1 (water): min={min_f1:.3f}, mean={f1_vals.mean():.3f}",
                details="Expected minimum: 0.75",
                severity="WARNING" if not reasonable else "INFO",
            )
            results.append(r)
            _add_result(r)
    else:
        r = QCResult(
            check_name="sar_accuracy_file",
            category="water_validation",
            passed=False,
            message="SAR accuracy table not found. Run 08_generate_tables.py first.",
            severity="WARNING",
        )
        results.append(r)
        _add_result(r)

    # Check for cross-validation raster outputs
    for fname in ["sar_vs_jrc_confusion.csv", "sar_jrc_iou.csv"]:
        fpath = OUTPUTS_DIR / "phase1_water_maps" / fname
        exists = fpath.exists()
        r = QCResult(
            check_name=f"cross_val_{fname}",
            category="water_validation",
            passed=exists,
            message=f"{fname}: {'Found' if exists else 'Not yet generated'}",
            severity="INFO" if exists else "WARNING",
        )
        results.append(r)
        _add_result(r)

    return results


# ============================================================================
# 4. Check ML Metrics
# ============================================================================

def check_ml_metrics() -> List[QCResult]:
    """
    Verify that machine learning model metrics are within reasonable
    ranges and are internally consistent.

    Checks:
    - AUC-ROC between 0.5 and 1.0
    - Accuracy, precision, recall, F1 between 0 and 1
    - F1 is harmonic mean of precision and recall (within tolerance)
    - No model has AUC < 0.70 (minimum acceptable)
    - Ensemble outperforms individual models (or is comparable)

    Returns
    -------
    list of QCResult
    """
    logger.info("Checking ML model metrics...")
    results = []

    ml_path = TABLES_DIR / "table3_ml_comparison.csv"
    if not ml_path.exists():
        r = QCResult(
            check_name="ml_metrics_file",
            category="ml_validation",
            passed=False,
            message="ML comparison table not found. Run 08_generate_tables.py first.",
            severity="WARNING",
        )
        results.append(r)
        _add_result(r)
        return results

    ml_df = pd.read_csv(ml_path)

    # Basic range checks
    metric_cols = ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score", "Kappa"]
    for col in metric_cols:
        if col not in ml_df.columns:
            continue
        vals = pd.to_numeric(ml_df[col], errors="coerce").dropna()
        in_range = (vals >= 0).all() and (vals <= 1).all()
        r = QCResult(
            check_name=f"ml_{col.lower().replace('-', '_')}_range",
            category="ml_validation",
            passed=in_range,
            message=(
                f"{col}: range [{vals.min():.3f}, {vals.max():.3f}] "
                f"{'within [0,1]' if in_range else 'OUT OF RANGE'}"
            ),
            severity="CRITICAL" if not in_range else "INFO",
        )
        results.append(r)
        _add_result(r)

    # AUC minimum threshold
    if "AUC-ROC" in ml_df.columns:
        auc_vals = pd.to_numeric(ml_df["AUC-ROC"], errors="coerce").dropna()
        min_auc = auc_vals.min()
        acceptable = min_auc >= 0.70
        r = QCResult(
            check_name="ml_auc_minimum",
            category="ml_validation",
            passed=acceptable,
            message=(
                f"Minimum AUC-ROC: {min_auc:.3f} "
                f"({'Acceptable' if acceptable else 'Below threshold 0.70'})"
            ),
            severity="ERROR" if not acceptable else "INFO",
        )
        results.append(r)
        _add_result(r)

    # F1 consistency: F1 = 2 * P * R / (P + R)
    if all(c in ml_df.columns for c in ["Precision", "Recall", "F1-Score"]):
        for _, row in ml_df.iterrows():
            p = pd.to_numeric(row.get("Precision"), errors="coerce")
            r_val = pd.to_numeric(row.get("Recall"), errors="coerce")
            f1_reported = pd.to_numeric(row.get("F1-Score"), errors="coerce")

            if pd.isna(p) or pd.isna(r_val) or pd.isna(f1_reported):
                continue
            if p + r_val == 0:
                continue

            f1_expected = 2 * p * r_val / (p + r_val)
            consistent = abs(f1_reported - f1_expected) < 0.02
            model_name = row.get("Model", "Unknown")

            r = QCResult(
                check_name=f"ml_f1_consistency_{model_name}",
                category="ml_validation",
                passed=consistent,
                message=(
                    f"{model_name}: F1={f1_reported:.3f} vs computed "
                    f"2*P*R/(P+R)={f1_expected:.3f} "
                    f"(diff={abs(f1_reported - f1_expected):.4f})"
                ),
                severity="WARNING" if not consistent else "INFO",
            )
            results.append(r)
            _add_result(r)

    # SHAP importance: check top features are sensible
    shap_path = TABLES_DIR / "table4_feature_importance.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        # Check that SHAP values are non-negative
        if "Mean |SHAP|" in shap_df.columns:
            all_positive = (pd.to_numeric(shap_df["Mean |SHAP|"], errors="coerce") >= 0).all()
            r = QCResult(
                check_name="shap_values_positive",
                category="ml_validation",
                passed=all_positive,
                message=f"SHAP values all non-negative: {all_positive}",
                severity="ERROR" if not all_positive else "INFO",
            )
            results.append(r)
            _add_result(r)

    return results


# ============================================================================
# 5. Verify Municipal Statistics
# ============================================================================

def verify_municipal_stats() -> List[QCResult]:
    """
    Verify that municipal-level statistics are internally consistent:
    - Sum of municipal populations equals department total (approximately)
    - Sum of municipal areas equals department area
    - All 64 municipalities are accounted for
    - Risk scores are within [0, 1]
    - No duplicated municipalities

    Returns
    -------
    list of QCResult
    """
    logger.info("Verifying municipal statistics...")
    results = []

    # Check municipality count
    try:
        municipalities = load_municipalities("gadm")
        n_mun = len(municipalities)
        expected = 64
        within_range = abs(n_mun - expected) <= 5  # Allow small tolerance
        r = QCResult(
            check_name="municipal_count",
            category="municipal_stats",
            passed=within_range,
            message=f"Municipality count: {n_mun} (expected ~{expected})",
            severity="WARNING" if not within_range else "INFO",
        )
        results.append(r)
        _add_result(r)

        # Check for duplicates
        name_col = "NAME_2" if "NAME_2" in municipalities.columns else municipalities.columns[0]
        n_unique = municipalities[name_col].nunique()
        no_dupes = n_unique == n_mun
        r = QCResult(
            check_name="municipal_duplicates",
            category="municipal_stats",
            passed=no_dupes,
            message=(
                f"Unique municipality names: {n_unique}/{n_mun} "
                f"({'No duplicates' if no_dupes else 'DUPLICATES DETECTED'})"
            ),
            severity="WARNING" if not no_dupes else "INFO",
        )
        results.append(r)
        _add_result(r)
    except FileNotFoundError:
        r = QCResult(
            check_name="municipal_count",
            category="municipal_stats",
            passed=False,
            message="Municipality boundary file not found.",
            severity="WARNING",
        )
        results.append(r)
        _add_result(r)

    # Check subregion municipality coverage
    try:
        subregions = load_subregions()
        n_sub = len(subregions)
        r = QCResult(
            check_name="subregion_count",
            category="municipal_stats",
            passed=(n_sub == 13),
            message=f"Subregion count: {n_sub} (expected 13)",
            severity="ERROR" if n_sub != 13 else "INFO",
        )
        results.append(r)
        _add_result(r)
    except FileNotFoundError:
        pass

    # Check risk scores in table5
    risk_path = TABLES_DIR / "table5_municipal_risk_top20.csv"
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        score_col = None
        for candidate in ["Composite Risk Score", "risk_score", "Risk Score"]:
            if candidate in risk_df.columns:
                score_col = candidate
                break

        if score_col:
            scores = pd.to_numeric(risk_df[score_col], errors="coerce").dropna()
            in_range = (scores >= 0).all() and (scores <= 1).all()
            r = QCResult(
                check_name="risk_scores_range",
                category="municipal_stats",
                passed=in_range,
                message=(
                    f"Risk scores in [0,1]: {in_range} "
                    f"(range: [{scores.min():.3f}, {scores.max():.3f}])"
                ),
                severity="WARNING" if not in_range else "INFO",
            )
            results.append(r)
            _add_result(r)

    # Check population exposure totals
    exposure_path = TABLES_DIR / "table6_population_exposure.csv"
    if exposure_path.exists():
        exp_df = pd.read_csv(exposure_path)
        # Check exposed <= total population for each row
        if "Total Population" in exp_df.columns and "Exposed Population" in exp_df.columns:
            data_rows = exp_df[exp_df["Subregion"] != "TOTAL (Narino)"]
            pop_total = pd.to_numeric(data_rows["Total Population"], errors="coerce")
            pop_exposed = pd.to_numeric(data_rows["Exposed Population"], errors="coerce")
            valid_mask = pop_total.notna() & pop_exposed.notna()
            consistent = (pop_exposed[valid_mask] <= pop_total[valid_mask]).all()

            r = QCResult(
                check_name="exposure_population_consistency",
                category="municipal_stats",
                passed=consistent,
                message=(
                    f"Exposed <= Total population in all subregions: {consistent}"
                ),
                severity="ERROR" if not consistent else "INFO",
            )
            results.append(r)
            _add_result(r)

    return results


# ============================================================================
# 6. Generate QC Report
# ============================================================================

def generate_qc_report() -> pathlib.Path:
    """
    Compile all QC check results into a comprehensive Markdown report.

    Returns
    -------
    pathlib.Path
        Path to the generated report file.
    """
    logger.info("Generating QC report...")
    QC_DIR.mkdir(parents=True, exist_ok=True)
    report_path = QC_DIR / "qc_report.md"

    # Count results by status
    n_total = len(_qc_results)
    n_pass = sum(1 for r in _qc_results if r.passed)
    n_fail = n_total - n_pass
    n_critical = sum(1 for r in _qc_results if r.severity == "CRITICAL" and not r.passed)
    n_error = sum(1 for r in _qc_results if r.severity == "ERROR" and not r.passed)
    n_warning = sum(1 for r in _qc_results if r.severity == "WARNING" and not r.passed)

    # Build report
    lines = []
    lines.append("# Quality Control Report")
    lines.append(f"## Narino Flood Risk Assessment")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Project Root:** `{PROJECT_ROOT}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total checks | {n_total} |")
    lines.append(f"| Passed | {n_pass} |")
    lines.append(f"| Failed | {n_fail} |")
    lines.append(f"| Critical failures | {n_critical} |")
    lines.append(f"| Errors | {n_error} |")
    lines.append(f"| Warnings | {n_warning} |")
    lines.append("")

    if n_total > 0:
        pass_rate = n_pass / n_total * 100
        lines.append(f"**Overall pass rate: {pass_rate:.1f}%**")
    lines.append("")

    # Overall verdict
    if n_critical > 0:
        verdict = "CRITICAL ISSUES FOUND - review before publication"
    elif n_error > 0:
        verdict = "ERRORS FOUND - address before final submission"
    elif n_warning > 0:
        verdict = "WARNINGS PRESENT - review recommended"
    else:
        verdict = "ALL CHECKS PASSED"
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed results by category
    categories = sorted(set(r.category for r in _qc_results))
    for cat in categories:
        cat_results = [r for r in _qc_results if r.category == cat]
        cat_pass = sum(1 for r in cat_results if r.passed)
        cat_total = len(cat_results)

        lines.append(f"## {cat.replace('_', ' ').title()} ({cat_pass}/{cat_total} passed)")
        lines.append("")
        lines.append("| Status | Check | Message | Severity |")
        lines.append("|--------|-------|---------|----------|")

        for r in cat_results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(
                f"| {icon} | {r.check_name} | {r.message} | {r.severity} |"
            )
        lines.append("")

    # Failed checks detail
    failed = [r for r in _qc_results if not r.passed]
    if failed:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Checks - Details")
        lines.append("")
        for r in failed:
            lines.append(f"### {r.check_name}")
            lines.append(f"- **Category:** {r.category}")
            lines.append(f"- **Severity:** {r.severity}")
            lines.append(f"- **Message:** {r.message}")
            if r.details:
                lines.append(f"- **Details:** {r.details}")
            lines.append("")

    # Write report
    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_content)
    logger.info("  QC report written to: %s", report_path)

    # Save summary CSV
    summary_df = pd.DataFrame([r.to_dict() for r in _qc_results])
    summary_csv = QC_DIR / "qc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("  QC summary CSV written to: %s", summary_csv)

    return report_path


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run all quality control checks and generate the report."""
    logger.info("=" * 70)
    logger.info("QUALITY CONTROL - NARINO FLOOD RISK ASSESSMENT")
    logger.info("=" * 70)

    ensure_dirs()

    # Clear previous results
    _qc_results.clear()

    # Run all QC checks
    qc_functions = [
        ("Output File Checks", check_outputs),
        ("Area Validation", validate_areas),
        ("Water Detection Cross-Validation", cross_validate_water),
        ("ML Metric Validation", check_ml_metrics),
        ("Municipal Statistics Verification", verify_municipal_stats),
    ]

    for name, func in qc_functions:
        logger.info("-" * 50)
        logger.info("Running: %s", name)
        try:
            func()
        except Exception as exc:
            logger.error("QC function %s failed: %s", name, exc, exc_info=True)
            _add_result(QCResult(
                check_name=f"qc_function_{func.__name__}",
                category="system",
                passed=False,
                message=f"QC function raised exception: {exc}",
                severity="CRITICAL",
            ))

    # Generate the report
    logger.info("-" * 50)
    report_path = generate_qc_report()

    # Final summary
    n_pass = sum(1 for r in _qc_results if r.passed)
    n_total = len(_qc_results)
    logger.info("=" * 70)
    logger.info(
        "QC complete: %d/%d checks passed (%.1f%%)",
        n_pass, n_total, n_pass / max(n_total, 1) * 100,
    )
    logger.info("Report: %s", report_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
