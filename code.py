from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import least_squares, minimize, nnls
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from scipy.stats import normaltest, spearmanr
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_NAME = "TRACE: Transition-Regime Attribution and Calibrated Extrapolation"
Q_MIN, Q_MAX = 1e-4, 5.0
F_MIN, F_MAX = 1e-4, 1.5
H_MIN, H_MAX = 1e-4, 1.5
RAW_CLIMATE_SOURCE = ["clim_tmean_c", "clim_vpd_mean_kpa", "clim_prcp_annualized_mm", "clim_deficit_annualized_mm"]
RAW_CLIMATE_MODEL = ["clim_tmean_c", "clim_vpd_mean_kpa", "log_prcp", "log_deficit"]
BASE_FEATURE_ORDER = ["u0_log", "v0_log", "dt_years", "log_dt", "target_year", "decimalLatitude", "decimalLongitude", "young_baseline_area_precision", "mature_baseline_area_precision", "young_baseline_area_missing", "mature_baseline_area_missing", "zero_u0", "zero_v0"]
DEFAULT_FEATURE_WEIGHT_MAP = {name: 1.0 for name in [*BASE_FEATURE_ORDER, "climate_PC1", "climate_PC2"]}
REQUIRED = [
    "transition_id",
    "siteID",
    "plotID",
    "dt_years",
    "event_year",
    "decimalLatitude",
    "decimalLongitude",
    "u0_kha",
    "v0_kha",
    "u1_kha",
    "v1_kha",
    "young_area_precision",
    "mature_area_precision",
    "area_young_m2",
    "area_mature_m2",
    "young_count_t0",
    "mature_count_t0",
    "young_count_t1",
    "mature_count_t1",
    *RAW_CLIMATE_SOURCE,
]

@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_scaffold: bool = True
    use_teacher: bool = True
    use_climate: bool = True
    use_coordinates: bool = True
    use_precision_features: bool = True
    use_precision_weights: bool = False
    use_variance_weights: bool = True
    use_occurrence: bool = True

@dataclass(frozen=True)
class HyperParameters:
    gate: float = 0.25
    ridge_scale: float = 1.0
    distillation_weight: float = 0.75
    occurrence_gate: float = 1.0
    gamma_scale: float = 1.0
    kernel_alpha: float = 0.50

@dataclass
class Configuration:
    outer_folds: int = 3
    inner_folds: int = 3
    calibration_fraction: float = 0.35
    stochastic_folds: int = 3
    interval: float = 0.90
    conformal_mode: str = "site_quantile"
    conformal_site_quantile: float = 0.90
    cluster_bootstrap: int = 1000
    paired_bootstrap: int = 1000
    regime_bootstrap: int = 1000
    signflip_permutations: int = 5000
    residual_permutations: int = 2000
    mc_draws: int = 512
    log_diffusion_draws: int = 512
    positive_diffusion_draws: int = 512
    process_trim_quantile: float = 0.975
    integration_steps: int = 24
    log_diffusion_steps: int = 32
    positive_diffusion_steps: int = 32
    positive_diffusion_demographic_fraction: float = 0.50
    positive_diffusion_scale: float = 1.00
    scaffold_multistart: int = 4
    scaffold_max_nfev: int = 300
    scaffold_bootstrap_multistart: int = 1
    scaffold_bootstrap_max_nfev: int = 160
    ode_bootstrap: int = 200
    ode_sensitivity_rows: int = 600
    ode_profile_points: int = 7
    ode_profile_max_nfev: int = 100
    ode_profile_rows: int = 800
    derivative_bootstrap: int = 80
    derivative_reference_rows: int = 256
    teacher_folds: int = 3
    teacher_trees: int = 240
    teacher_max_depth: int = 9
    teacher_min_leaf: int = 6
    occurrence_folds: int = 4
    occurrence_trees: int = 360
    occurrence_min_leaf: int = 8
    probability_clip: float = 0.002
    rbf_centers: int = 128
    kernel_gammas: tuple[float, ...] = (0.01, 0.08)
    kernel_weights: tuple[float, ...] = (0.50, 0.50)
    young_kernel_ridge: float = 0.15
    mature_kernel_ridge: float = 0.80
    kernel_jitter: float = 1e-8
    feature_weight_map: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHT_MAP))
    tune_model: bool = True
    tune_benchmarks: bool = True
    gate_grid: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
    ridge_scale_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    distillation_weight_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    occurrence_gate_grid: tuple[float, ...] = (0.0, 0.50, 1.0)
    gamma_scale_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    kernel_alpha_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    tuning_zero_weight: float = 0.20
    variance_weight_folds: int = 3
    variance_count_offset: float = 0.50
    variance_weight_lower_quantile: float = 0.02
    variance_weight_upper_quantile: float = 0.98
    support_change_tolerance: float = 0.20
    tuning_refinement: bool = True
    repeated_holdouts: int = 30
    repeated_test_fraction: float = 0.25
    surrogate_splits: int = 20
    surrogate_top_terms: int = 24
    surrogate_max_iter: int = 10000
    log_diffusion_enabled: bool = True
    positive_diffusion_enabled: bool = True
    temporal_validation_enabled: bool = True
    temporal_cut_quantiles: tuple[float, ...] = (0.60, 0.75)
    spatiotemporal_repeats: int = 5
    spatiotemporal_test_fraction: float = 0.25
    minimum_temporal_train_sites: int = 8
    minimum_temporal_test_sites: int = 4
    state_log_cap: float = 10.0
    dpi: int = 300
    seed: int = 2026
    profile: str = "full"

    @classmethod
    def for_profile(cls, profile: str, **overrides: Any) -> "Configuration":
        profile = profile.lower()
        if profile not in {"smoke", "standard", "full"}:
            raise ValueError("profile must be smoke, standard, or full")
        c = cls(profile=profile)
        if profile == "smoke":
            c.outer_folds = 2
            c.inner_folds = 2
            c.cluster_bootstrap = 20
            c.paired_bootstrap = 20
            c.regime_bootstrap = 20
            c.signflip_permutations = 100
            c.residual_permutations = 100
            c.mc_draws = 16
            c.log_diffusion_draws = 32
            c.positive_diffusion_draws = 32
            c.integration_steps = 10
            c.log_diffusion_steps = 12
            c.positive_diffusion_steps = 12
            c.scaffold_multistart = 1
            c.scaffold_max_nfev = 10
            c.scaffold_bootstrap_max_nfev = 50
            c.ode_bootstrap = 0
            c.ode_sensitivity_rows = 120
            c.ode_profile_points = 3
            c.ode_profile_max_nfev = 5
            c.ode_profile_rows = 100
            c.derivative_bootstrap = 0
            c.derivative_reference_rows = 64
            c.teacher_folds = 2
            c.teacher_trees = 8
            c.occurrence_folds = 2
            c.occurrence_trees = 12
            c.rbf_centers = 12
            c.gate_grid = (0.0, 0.5, 1.0)
            c.ridge_scale_grid = (1.0,)
            c.distillation_weight_grid = (0.5, 1.0)
            c.occurrence_gate_grid = (0.0, 1.0)
            c.gamma_scale_grid = (1.0,)
            c.kernel_alpha_grid = (0.5,)
            c.variance_weight_folds = 2
            c.tuning_refinement = False
            c.repeated_holdouts = 0
            c.spatiotemporal_repeats = 0
            c.temporal_cut_quantiles = (0.70,)
            c.surrogate_splits = 1
            c.tune_benchmarks = False
            c.log_diffusion_enabled = False
            c.positive_diffusion_enabled = False
            c.temporal_validation_enabled = False
            c.dpi = 100
        elif profile == "standard":
            c.cluster_bootstrap = 400
            c.paired_bootstrap = 400
            c.regime_bootstrap = 400
            c.signflip_permutations = 1500
            c.residual_permutations = 800
            c.mc_draws = 256
            c.log_diffusion_draws = 256
            c.positive_diffusion_draws = 256
            c.integration_steps = 18
            c.log_diffusion_steps = 24
            c.positive_diffusion_steps = 24
            c.scaffold_multistart = 2
            c.scaffold_max_nfev = 180
            c.ode_bootstrap = 50
            c.ode_sensitivity_rows = 350
            c.ode_profile_points = 5
            c.ode_profile_max_nfev = 60
            c.ode_profile_rows = 350
            c.derivative_bootstrap = 10
            c.derivative_reference_rows = 160
            c.teacher_trees = 120
            c.occurrence_trees = 180
            c.rbf_centers = 96
            c.gate_grid = (0.0, 0.25, 0.5, 1.0)
            c.ridge_scale_grid = (0.5, 1.0, 2.0)
            c.distillation_weight_grid = (0.5, 0.75, 1.0)
            c.occurrence_gate_grid = (0.0, 0.5, 1.0)
            c.gamma_scale_grid = (0.5, 1.0, 2.0)
            c.kernel_alpha_grid = (0.25, 0.5, 0.75)
            c.repeated_holdouts = 10
            c.spatiotemporal_repeats = 3
            c.surrogate_splits = 10
            c.dpi = 220
        for key, value in overrides.items():
            if value is not None:
                setattr(c, key, value)
        return c

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

def stable_int_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 1000000000

def finite_quantile(values: np.ndarray, probability: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    rank = min(max(int(math.ceil((x.size + 1) * probability)), 1), x.size)
    return float(np.partition(x, rank - 1)[rank - 1])

def ensure_probability(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1), got {value}")

def safe_spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    result = spearmanr(y, prediction, nan_policy="omit")
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value) if np.isfinite(value) else float("nan")

def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, interval: float) -> float:
    alpha = 1.0 - interval
    width = upper - lower
    penalty_low = (2.0 / alpha) * (lower - y) * (y < lower)
    penalty_high = (2.0 / alpha) * (y - upper) * (y > upper)
    return float(np.mean(width + penalty_low + penalty_high))

def crps_samples(samples: np.ndarray, observations: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(samples, dtype=float), axis=0)
    y = np.asarray(observations, dtype=float)
    draws = x.shape[0]
    first = np.mean(np.abs(x - y[None, :]), axis=0)
    weights = (2 * np.arange(1, draws + 1) - draws - 1).reshape(-1, 1)
    second = np.sum(weights * x, axis=0) / (draws * draws)
    return first - second

def randomized_pit(samples: np.ndarray, observations: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(observations, dtype=float)
    less = np.mean(samples < y[None, :], axis=0)
    equal = np.mean(np.isclose(samples, y[None, :], atol=1e-12, rtol=0.0), axis=0)
    return np.clip(less + rng.random(len(y)) * equal, 0.0, 1.0)

def validate_no_group_overlap(*frames: pd.DataFrame) -> bool:
    groups = [set(frame.siteID.astype(str)) for frame in frames]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if groups[left] & groups[right]:
                return False
    return True

def infer_baseline_sampling_area(data: pd.DataFrame, stage: str) -> tuple[np.ndarray, np.ndarray]:
    explicit = f"area_{stage}_t0_m2"
    endpoint = f"area_{stage}_m2"
    count = f"{stage}_count_t0"
    previous_density = f"previous_{stage}_per_ha"
    area = np.full(len(data), np.nan, dtype=float)
    source = np.full(len(data), "unresolved", dtype=object)
    if explicit in data.columns:
        values = pd.to_numeric(data[explicit], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & (values > 0)
        area[valid] = values[valid]
        source[valid] = "explicit_t0_area"
    if {"previous_event_date", "event_date", endpoint}.issubset(data.columns):
        event = pd.to_datetime(data.event_date, errors="coerce")
        previous = pd.to_datetime(data.previous_event_date, errors="coerce")
        lookup = {}
        endpoint_area = pd.to_numeric(data[endpoint], errors="coerce").to_numpy(float)
        for i in range(len(data)):
            if pd.notna(event.iloc[i]) and np.isfinite(endpoint_area[i]) and endpoint_area[i] > 0:
                lookup[(str(data.siteID.iloc[i]), str(data.plotID.iloc[i]), event.iloc[i])] = endpoint_area[i]
        unresolved = ~np.isfinite(area)
        for i in np.flatnonzero(unresolved):
            key = (str(data.siteID.iloc[i]), str(data.plotID.iloc[i]), previous.iloc[i])
            if pd.notna(previous.iloc[i]) and key in lookup:
                area[i] = lookup[key]
                source[i] = "linked_previous_transition"
    if previous_density in data.columns:
        counts = pd.to_numeric(data[count], errors="coerce").to_numpy(float)
        density = pd.to_numeric(data[previous_density], errors="coerce").to_numpy(float)
        valid = (~np.isfinite(area)) & (counts > 0) & (density > 0) & np.isfinite(counts) & np.isfinite(density)
        area[valid] = counts[valid] / density[valid] * 10000.0
        source[valid] = "inferred_from_positive_count_density"
    return area, source

def data_role_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    identifiers = {"transition_id", "site_index", "plot_index", "siteID", "plotID", "eventID"}
    baseline = {"previous_event_date", "previous_event_year", "u0_kha", "v0_kha", "young_count_t0", "mature_count_t0", "previous_young_per_ha", "previous_mature_per_ha", "decimalLatitude", "decimalLongitude", "plotType", "nlcdClass", "area_young_t0_m2_reconstructed", "area_mature_t0_m2_reconstructed", "young_baseline_area_precision", "mature_baseline_area_precision", "young_baseline_area_missing", "mature_baseline_area_missing"}
    endpoint = {"event_date", "event_year", "u1_kha", "v1_kha", "young_count_t1", "mature_count_t1", "young_per_ha", "mature_per_ha", "area_young_m2", "area_mature_m2"}
    post = {"young_log_growth_rate", "mature_log_growth_rate", "death_events", "maturation_events", "death_fraction_interval", "maturation_fraction_interval", "death_hazard_annual", "maturation_hazard_annual"}
    realized = set(column for column in data.columns if column.startswith("clim_")) | {"climate_days", "climate_expected_days", "climate_coverage", "log_prcp", "log_deficit"}
    globally_standardized = {column for column in data.columns if column.endswith("_z")}
    baseline_derived = {"u0_log", "v0_log", "zero_u0", "zero_v0"}
    derived = {"u1_log", "v1_log", "delta_u_log", "delta_v_log", "positive_u1", "positive_v1", "row_id", "sequence_index_plot", "n_transitions_plot", "t0_date", "t1_date", "t0_year", "t1_year", "midpoint_year"}
    scheduled = {"target_year", "dt_years", "log_dt"}
    rows = []
    for column in data.columns:
        if column in identifiers:
            role = "identifier"
            availability = "metadata"
            model_allowed = False
        elif column in baseline:
            role = "baseline"
            availability = "t0"
            model_allowed = column in BASE_FEATURE_ORDER or column in {"u0_kha", "v0_kha"}
        elif column in endpoint:
            role = "endpoint_observation"
            availability = "t1"
            model_allowed = False
        elif column in scheduled:
            role = "scheduled_horizon"
            availability = "t0_known_horizon"
            model_allowed = True
        elif column in baseline_derived:
            role = "baseline_derived"
            availability = "t0_derived"
            model_allowed = True
        elif column in post:
            role = "post_interval_audit"
            availability = "post_t1"
            model_allowed = False
        elif column in globally_standardized:
            role = "global_standardization_audit"
            availability = "derived_full_data"
            model_allowed = False
        elif column in realized:
            role = "realized_interval_covariate"
            availability = "t0_to_t1"
            model_allowed = column in RAW_CLIMATE_SOURCE or column in RAW_CLIMATE_MODEL
        elif column in derived:
            role = "derived"
            availability = "derived"
            model_allowed = column in BASE_FEATURE_ORDER
        elif column == "n_transitions_site":
            role = "sampling_metadata"
            availability = "full_dataset"
            model_allowed = False
        else:
            role = "auxiliary"
            availability = "unspecified"
            model_allowed = False
        rows.append({"column": column, "role": role, "availability": availability, "model_allowed": bool(model_allowed), "dtype": str(data[column].dtype)})
    return pd.DataFrame(rows)


def predictor_availability_audit(data: pd.DataFrame, spec: ModelSpec, feature_names: Sequence[str]) -> pd.DataFrame:
    dictionary = data_role_dictionary(data).set_index("column")
    rows = []
    permitted_availability = {"t0", "t0_derived", "t0_known_horizon", "t0_to_t1"}
    for feature in feature_names:
        if feature in {"climate_PC1", "climate_PC2"}:
            role = "realized_interval_covariate"
            availability = "t0_to_t1"
            available_for_target_estimation = bool(spec.use_climate)
            allowed_in_spec = bool(spec.use_climate)
        elif feature in dictionary.index:
            role = str(dictionary.at[feature, "role"])
            availability = str(dictionary.at[feature, "availability"])
            available_for_target_estimation = availability in permitted_availability
            allowed_in_spec = bool(dictionary.at[feature, "model_allowed"]) and available_for_target_estimation
        else:
            role = "unknown"
            availability = "unknown"
            available_for_target_estimation = False
            allowed_in_spec = False
        rows.append({"model": spec.name, "feature": feature, "role": role, "availability": availability, "available_for_target_estimation": available_for_target_estimation, "allowed_in_spec": allowed_in_spec})
    audit = pd.DataFrame(rows)
    invalid = audit[~audit.allowed_in_spec]
    if not invalid.empty:
        raise RuntimeError("Predictor availability violation: " + ", ".join(invalid.feature.astype(str)))
    forbidden_roles = {"endpoint_observation", "post_interval_audit", "global_standardization_audit"}
    forbidden = audit[audit.role.isin(forbidden_roles)]
    if not forbidden.empty:
        raise RuntimeError("Endpoint or post-outcome information entered the predictor set: " + ", ".join(forbidden.feature.astype(str)))
    return audit

def data_quality_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ["young", "mature"]:
        baseline = data[f"area_{stage}_t0_m2_reconstructed"].to_numpy(float)
        endpoint = data[f"area_{stage}_m2"].to_numpy(float)
        known = np.isfinite(baseline)
        changed = known & (np.abs(baseline - endpoint) > 1e-8)
        zero = data[f"{stage}_count_t0"].to_numpy(float) == 0
        rows.extend([
            {"item": f"{stage}_baseline_area_known_fraction", "value": float(np.mean(known))},
            {"item": f"{stage}_baseline_area_known_among_t0_zero_fraction", "value": float(np.mean(known[zero])) if zero.any() else np.nan},
            {"item": f"{stage}_known_area_changed_fraction", "value": float(np.mean(changed[known])) if known.any() else np.nan},
            {"item": f"{stage}_endpoint_area_min_m2", "value": float(np.nanmin(endpoint))},
            {"item": f"{stage}_endpoint_area_max_m2", "value": float(np.nanmax(endpoint))},
        ])
    rows.extend([
        {"item": "transitions", "value": float(len(data))},
        {"item": "sites", "value": float(data.siteID.nunique())},
        {"item": "plots", "value": float(data.plotID.nunique())},
        {"item": "plots_with_multiple_transitions", "value": float((data.groupby(["siteID", "plotID"]).size() > 1).sum())},
        {"item": "global_z_columns_blacklisted", "value": float(sum(column.endswith("_z") for column in data.columns))},
        {"item": "post_outcome_columns_blacklisted", "value": float(sum(column in {"young_log_growth_rate", "mature_log_growth_rate", "death_events", "maturation_events", "death_fraction_interval", "maturation_fraction_interval", "death_hazard_annual", "maturation_hazard_annual"} for column in data.columns))},
    ])
    return pd.DataFrame(rows)

def missingness_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        missing = int(data[column].isna().sum())
        if missing > 0:
            rows.append({"column": column, "missing": missing, "missing_fraction": float(missing / len(data))})
    return pd.DataFrame(rows).sort_values(["missing_fraction", "column"], ascending=[False, True]).reset_index(drop=True) if rows else pd.DataFrame(columns=["column", "missing", "missing_fraction"])

def prepare_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    identifiers = {"transition_id", "siteID", "plotID"}
    for column in [c for c in REQUIRED if c not in identifiers]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED).copy()
    data = data[
        (data.dt_years > 0)
        & (data.u0_kha >= 0)
        & (data.v0_kha >= 0)
        & (data.u1_kha >= 0)
        & (data.v1_kha >= 0)
        & (data.area_young_m2 > 0)
        & (data.area_mature_m2 > 0)
    ].copy()
    data = data.sort_values(["transition_id", "siteID", "plotID"]).drop_duplicates("transition_id").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid transitions remain after quality control")
    if data.siteID.nunique() < 4:
        raise ValueError("At least four independent sites are required")
    data["t0_date"] = pd.to_datetime(data["previous_event_date"], errors="coerce") if "previous_event_date" in data.columns else pd.NaT
    data["t1_date"] = pd.to_datetime(data["event_date"], errors="coerce") if "event_date" in data.columns else pd.NaT
    data["t0_year"] = data.t0_date.dt.year.astype(float) if data.t0_date.notna().any() else pd.to_numeric(data.get("previous_event_year", np.nan), errors="coerce")
    data["t1_year"] = data.t1_date.dt.year.astype(float) if data.t1_date.notna().any() else pd.to_numeric(data.event_year, errors="coerce")
    data["target_year"] = data.t0_year.to_numpy(float) + data.dt_years.to_numpy(float)
    data["midpoint_year"] = 0.5 * (data.t0_year.to_numpy(float) + data.t1_year.to_numpy(float))
    for stage, scale in [("young", 200.0), ("mature", 800.0)]:
        area, source = infer_baseline_sampling_area(data, stage)
        data[f"area_{stage}_t0_m2_reconstructed"] = area
        data[f"area_{stage}_t0_source"] = source
        data[f"{stage}_baseline_area_missing"] = (~np.isfinite(area)).astype(float)
        data[f"{stage}_baseline_area_precision"] = np.sqrt(np.maximum(area, 0.0) / scale)
        data[f"{stage}_area_changed_known"] = (np.isfinite(area) & (np.abs(area - data[f"area_{stage}_m2"].to_numpy(float)) > 1e-8)).astype(int)
    data["u0_log"] = np.log1p(data.u0_kha)
    data["v0_log"] = np.log1p(data.v0_kha)
    data["u1_log"] = np.log1p(data.u1_kha)
    data["v1_log"] = np.log1p(data.v1_kha)
    data["delta_u_log"] = data.u1_log - data.u0_log
    data["delta_v_log"] = data.v1_log - data.v0_log
    data["log_dt"] = np.log(data.dt_years)
    data["log_prcp"] = np.log1p(data.clim_prcp_annualized_mm.clip(lower=0))
    data["log_deficit"] = np.log1p(data.clim_deficit_annualized_mm.clip(lower=0))
    data["zero_u0"] = (data.u0_kha == 0).astype(float)
    data["zero_v0"] = (data.v0_kha == 0).astype(float)
    data["positive_u1"] = (data.u1_kha > 0).astype(int)
    data["positive_v1"] = (data.v1_kha > 0).astype(int)
    order_columns = ["siteID", "plotID", "t0_date", "t1_date"] if data.t0_date.notna().any() else ["siteID", "plotID", "event_year"]
    ordered = data.sort_values(order_columns).copy()
    ordered["sequence_index_plot"] = ordered.groupby(["siteID", "plotID"]).cumcount() + 1
    ordered["n_transitions_plot"] = ordered.groupby(["siteID", "plotID"])["transition_id"].transform("size")
    data = data.join(ordered[["sequence_index_plot", "n_transitions_plot"]].sort_index())
    data["row_id"] = np.arange(len(data), dtype=int)
    return data

@dataclass
class ClimateTransform:
    scaler: StandardScaler
    pca: PCA
    orientation: np.ndarray

    @classmethod
    def fit(cls, data: pd.DataFrame, seed: int) -> "ClimateTransform":
        scaler = StandardScaler().fit(data[RAW_CLIMATE_MODEL])
        standardized = scaler.transform(data[RAW_CLIMATE_MODEL])
        pca = PCA(n_components=2, random_state=seed).fit(standardized)
        components = pca.components_.copy()
        orientation = np.ones(2, dtype=float)
        if components[0, 0] + components[0, 1] + components[0, 3] < 0:
            orientation[0] = -1.0
        if components[1, 2] * orientation[1] < 0:
            orientation[1] = -1.0
        return cls(scaler=scaler, pca=pca, orientation=orientation)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.pca.transform(self.scaler.transform(data[RAW_CLIMATE_MODEL]))
        return scores * self.orientation

    def loading_table(self, label: str) -> pd.DataFrame:
        rows = []
        oriented = self.pca.components_ * self.orientation[:, None]
        for index in range(2):
            for feature, loading in zip(RAW_CLIMATE_MODEL, oriented[index]):
                rows.append({
                    "fit": label,
                    "component": f"PC{index + 1}",
                    "feature": feature,
                    "loading": float(loading),
                    "explained_variance_ratio": float(self.pca.explained_variance_ratio_[index]),
                    "cumulative_explained_variance": float(self.pca.explained_variance_ratio_[: index + 1].sum()),
                })
        return pd.DataFrame(rows)

@dataclass
class FeatureTransform:
    feature_names: list[str]
    climate: ClimateTransform | None
    scaler: StandardScaler
    weights: np.ndarray
    impute_values: np.ndarray

    @staticmethod
    def selected_features(spec: ModelSpec) -> list[str]:
        selected = []
        precision_features = {"young_baseline_area_precision", "mature_baseline_area_precision", "young_baseline_area_missing", "mature_baseline_area_missing"}
        for feature in BASE_FEATURE_ORDER:
            if feature in {"decimalLatitude", "decimalLongitude"} and not spec.use_coordinates:
                continue
            if feature in precision_features and not spec.use_precision_features:
                continue
            selected.append(feature)
        if spec.use_climate:
            selected.extend(["climate_PC1", "climate_PC2"])
        return selected

    @classmethod
    def fit(cls, data: pd.DataFrame, spec: ModelSpec, config: Configuration, seed: int, climate_override: ClimateTransform | None = None) -> "FeatureTransform":
        climate = climate_override if spec.use_climate and climate_override is not None else ClimateTransform.fit(data, seed) if spec.use_climate else None
        names = cls.selected_features(spec)
        raw = cls._raw_matrix(data, names, climate)
        impute = np.zeros(raw.shape[1], dtype=float)
        for j in range(raw.shape[1]):
            finite = raw[:, j][np.isfinite(raw[:, j])]
            impute[j] = float(np.median(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(raw), raw, impute[None, :])
        scaler = StandardScaler().fit(filled)
        weights = np.array([config.feature_weight_map.get(name, 1.0) for name in names], dtype=float)
        return cls(feature_names=names, climate=climate, scaler=scaler, weights=weights, impute_values=impute)

    @staticmethod
    def _raw_matrix(data: pd.DataFrame, names: Sequence[str], climate: ClimateTransform | None) -> np.ndarray:
        columns = []
        climate_scores = climate.transform(data) if climate is not None else None
        for name in names:
            if name == "climate_PC1":
                if climate_scores is None:
                    raise RuntimeError("Climate transform unavailable")
                columns.append(climate_scores[:, 0])
            elif name == "climate_PC2":
                if climate_scores is None:
                    raise RuntimeError("Climate transform unavailable")
                columns.append(climate_scores[:, 1])
            else:
                columns.append(pd.to_numeric(data[name], errors="coerce").to_numpy(float))
        return np.column_stack(columns)

    def raw_matrix(self, data: pd.DataFrame) -> np.ndarray:
        raw = self._raw_matrix(data, self.feature_names, self.climate)
        return np.where(np.isfinite(raw), raw, self.impute_values[None, :])

    def transform(self, data: pd.DataFrame, weighted: bool = True) -> np.ndarray:
        raw = self.raw_matrix(data)
        standardized = np.clip(self.scaler.transform(raw), -6.0, 6.0)
        return standardized * self.weights if weighted else standardized

    def raw_derivative_scale(self) -> np.ndarray:
        return self.weights / np.maximum(self.scaler.scale_, 1e-12)

    def derivative_scale_matrix(self, data: pd.DataFrame) -> np.ndarray:
        raw_unfilled = self._raw_matrix(data, self.feature_names, self.climate)
        raw = np.where(np.isfinite(raw_unfilled), raw_unfilled, self.impute_values[None, :])
        standardized = (raw - self.scaler.mean_[None, :]) / np.maximum(self.scaler.scale_[None, :], 1e-12)
        active = (standardized > -6.0) & (standardized < 6.0) & np.isfinite(raw_unfilled)
        return active.astype(float) * self.raw_derivative_scale()[None, :]

@dataclass
class DemographicScaffold:
    climate: ClimateTransform | None
    parameters: np.ndarray
    integration_steps: int
    use_climate: bool
    fit_cost: float
    fit_success: bool
    active_bounds: np.ndarray
    nfev: int

    @staticmethod
    def rates(parameters: np.ndarray, climate_scores: np.ndarray) -> tuple[np.ndarray, ...]:
        q = Q_MIN + (Q_MAX - Q_MIN) * expit(parameters[0] + climate_scores @ parameters[1:3])
        maturation = F_MIN + (F_MAX - F_MIN) * expit(parameters[3] + climate_scores @ parameters[4:6])
        mortality = H_MIN + (H_MAX - H_MIN) * expit(parameters[6] + climate_scores @ parameters[7:9])
        rho, mu, a, b = np.exp(parameters[9:13])
        return q, maturation, mortality, rho, mu, a, b

    @staticmethod
    def drift(u: np.ndarray, v: np.ndarray, parameters: np.ndarray, climate_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q, maturation, mortality, rho, mu, a, b = DemographicScaffold.rates(parameters, climate_scores)
        du = q + rho * v - (mu + maturation) * u - a * u * u
        dv = maturation * u - mortality * v - b * v * v
        return du, dv

    @staticmethod
    def rate_derivatives(parameters: np.ndarray, climate_scores: np.ndarray) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q, maturation, mortality, rho, mu, a, b = DemographicScaffold.rates(parameters, climate_scores)
        n = len(climate_scores)
        p = len(parameters)
        dq = np.zeros((n, p), dtype=float)
        dF = np.zeros((n, p), dtype=float)
        dH = np.zeros((n, p), dtype=float)
        sq = expit(parameters[0] + climate_scores @ parameters[1:3])
        sF = expit(parameters[3] + climate_scores @ parameters[4:6])
        sH = expit(parameters[6] + climate_scores @ parameters[7:9])
        q_slope = (Q_MAX - Q_MIN) * sq * (1.0 - sq)
        F_slope = (F_MAX - F_MIN) * sF * (1.0 - sF)
        H_slope = (H_MAX - H_MIN) * sH * (1.0 - sH)
        dq[:, 0] = q_slope
        dq[:, 1] = q_slope * climate_scores[:, 0]
        dq[:, 2] = q_slope * climate_scores[:, 1]
        dF[:, 3] = F_slope
        dF[:, 4] = F_slope * climate_scores[:, 0]
        dF[:, 5] = F_slope * climate_scores[:, 1]
        dH[:, 6] = H_slope
        dH[:, 7] = H_slope * climate_scores[:, 0]
        dH[:, 8] = H_slope * climate_scores[:, 1]
        drho = np.zeros(p, dtype=float)
        dmu = np.zeros(p, dtype=float)
        da = np.zeros(p, dtype=float)
        db = np.zeros(p, dtype=float)
        drho[9] = rho
        dmu[10] = mu
        da[11] = a
        db[12] = b
        return (q, maturation, mortality, rho, mu, a, b), dq, dF, dH, drho, dmu, da, db

    @staticmethod
    def integrate_arrays_with_jacobian(data: pd.DataFrame, climate_scores: np.ndarray, parameters: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = np.maximum(data.u0_kha.to_numpy(float), 1e-10)
        v = np.maximum(data.v0_kha.to_numpy(float), 1e-10)
        dt = data.dt_years.to_numpy(float)
        step_count = max(1, int(steps))
        h = dt / step_count
        p = len(parameters)
        su = np.zeros((len(data), p), dtype=float)
        sv = np.zeros((len(data), p), dtype=float)
        rates, dq, dF, dH, drho, dmu, da, db = DemographicScaffold.rate_derivatives(parameters, climate_scores)
        q, maturation, mortality, rho, mu, a, b = rates
        for _ in range(step_count):
            fu = q + rho * v - (mu + maturation) * u - a * u * u
            fv = maturation * u - mortality * v - b * v * v
            bu = dq + v[:, None] * drho[None, :] - u[:, None] * (dmu[None, :] + dF) - (u * u)[:, None] * da[None, :]
            bv = u[:, None] * dF - v[:, None] * dH - (v * v)[:, None] * db[None, :]
            fu_u = -(mu + maturation) - 2.0 * a * u
            fu_v = rho
            fv_u = maturation
            fv_v = -mortality - 2.0 * b * v
            tfu = fu_u[:, None] * su + fu_v * sv + bu
            tfv = fv_u[:, None] * su + fv_v[:, None] * sv + bv
            ue = np.maximum(u + h * fu, 1e-10)
            ve = np.maximum(v + h * fv, 1e-10)
            active_u = (u + h * fu) > 1e-10
            active_v = (v + h * fv) > 1e-10
            sue = (su + h[:, None] * tfu) * active_u[:, None]
            sve = (sv + h[:, None] * tfv) * active_v[:, None]
            fue = q + rho * ve - (mu + maturation) * ue - a * ue * ue
            fve = maturation * ue - mortality * ve - b * ve * ve
            bue = dq + ve[:, None] * drho[None, :] - ue[:, None] * (dmu[None, :] + dF) - (ue * ue)[:, None] * da[None, :]
            bve = ue[:, None] * dF - ve[:, None] * dH - (ve * ve)[:, None] * db[None, :]
            fue_u = -(mu + maturation) - 2.0 * a * ue
            fue_v = rho
            fve_u = maturation
            fve_v = -mortality - 2.0 * b * ve
            tfue = fue_u[:, None] * sue + fue_v * sve + bue
            tfve = fve_u[:, None] * sue + fve_v[:, None] * sve + bve
            un = u + 0.5 * h * (fu + fue)
            vn = v + 0.5 * h * (fv + fve)
            sun = su + 0.5 * h[:, None] * (tfu + tfue)
            svn = sv + 0.5 * h[:, None] * (tfv + tfve)
            active_un = un > 1e-10
            active_vn = vn > 1e-10
            u = np.maximum(un, 1e-10)
            v = np.maximum(vn, 1e-10)
            su = sun * active_un[:, None]
            sv = svn * active_vn[:, None]
        return np.log1p(u), np.log1p(v), su / (1.0 + u)[:, None], sv / (1.0 + v)[:, None]

    @staticmethod
    def integrate_arrays(data: pd.DataFrame, climate_scores: np.ndarray, parameters: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        u = np.maximum(data.u0_kha.to_numpy(float), 1e-10)
        v = np.maximum(data.v0_kha.to_numpy(float), 1e-10)
        dt = data.dt_years.to_numpy(float)
        step_count = max(1, int(steps))
        h = dt / step_count
        for _ in range(step_count):
            du, dv = DemographicScaffold.drift(u, v, parameters, climate_scores)
            ue = np.maximum(u + h * du, 1e-10)
            ve = np.maximum(v + h * dv, 1e-10)
            due, dve = DemographicScaffold.drift(ue, ve, parameters, climate_scores)
            u = np.maximum(u + 0.5 * h * (du + due), 1e-10)
            v = np.maximum(v + 0.5 * h * (dv + dve), 1e-10)
        return np.log1p(u), np.log1p(v)

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        config: Configuration,
        seed: int,
        use_climate: bool = True,
        use_precision_weights: bool = True,
        climate_override: ClimateTransform | None = None,
        multistart_override: int | None = None,
        max_nfev_override: int | None = None,
        sample_weight_young: np.ndarray | None = None,
        sample_weight_mature: np.ndarray | None = None,
    ) -> "DemographicScaffold":
        rng = np.random.default_rng(seed)
        climate = climate_override if use_climate and climate_override is not None else ClimateTransform.fit(data, seed) if use_climate else None
        climate_scores = climate.transform(data) if climate is not None else np.zeros((len(data), 2))

        def bounded_logit(value: float, lower: float, upper: float) -> float:
            probability = np.clip((value - lower) / (upper - lower), 1e-5, 1 - 1e-5)
            return float(logit(probability))

        initial = np.array([
            bounded_logit(0.08, Q_MIN, Q_MAX), 0.0, 0.0,
            bounded_logit(0.08, F_MIN, F_MAX), 0.0, 0.0,
            bounded_logit(0.06, H_MIN, H_MAX), 0.0, 0.0,
            np.log(0.03), np.log(0.04), np.log(0.02), np.log(0.05),
        ])
        if sample_weight_young is not None:
            young_weight = np.sqrt(np.maximum(np.asarray(sample_weight_young, dtype=float), 1e-12))
        elif use_precision_weights:
            young_weight = np.sqrt(np.clip(data.young_area_precision.to_numpy(float), 0.5, 2.0))
        else:
            young_weight = np.ones(len(data))
        if sample_weight_mature is not None:
            mature_weight = np.sqrt(np.maximum(np.asarray(sample_weight_mature, dtype=float), 1e-12))
        elif use_precision_weights:
            mature_weight = np.sqrt(np.clip(data.mature_area_precision.to_numpy(float), 0.5, 2.0))
        else:
            mature_weight = np.ones(len(data))
        young_scale = np.std(data.delta_u_log.to_numpy(float)) + 0.10
        mature_scale = np.std(data.delta_v_log.to_numpy(float)) + 0.05

        cache_x = None
        cache_residual = None
        cache_jacobian = None

        def evaluate(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal cache_x, cache_residual, cache_jacobian
            if cache_x is not None and np.array_equal(parameters, cache_x):
                return cache_residual, cache_jacobian
            pred_u, pred_v, jac_u, jac_v = cls.integrate_arrays_with_jacobian(data, climate_scores, parameters, max(8, config.integration_steps - 2))
            data_residual = np.concatenate([
                (pred_u - data.u1_log.to_numpy(float)) * young_weight / young_scale,
                (pred_v - data.v1_log.to_numpy(float)) * mature_weight / mature_scale,
            ])
            data_jacobian = np.vstack([
                jac_u * (young_weight / young_scale)[:, None],
                jac_v * (mature_weight / mature_scale)[:, None],
            ])
            penalty_scale = 0.08 if use_climate else 2.0
            climate_indices = np.array([1, 2, 4, 5, 7, 8], dtype=int)
            climate_penalty = penalty_scale * parameters[climate_indices]
            climate_jacobian = np.zeros((len(climate_indices), len(parameters)), dtype=float)
            climate_jacobian[np.arange(len(climate_indices)), climate_indices] = penalty_scale
            positive_parameter_penalty = 0.05 * (parameters[9:13] - initial[9:13])
            positive_jacobian = np.zeros((4, len(parameters)), dtype=float)
            positive_jacobian[np.arange(4), np.arange(9, 13)] = 0.05
            cache_x = parameters.copy()
            cache_residual = np.concatenate([data_residual, climate_penalty, positive_parameter_penalty])
            cache_jacobian = np.vstack([data_jacobian, climate_jacobian, positive_jacobian])
            return cache_residual, cache_jacobian

        def residual(parameters: np.ndarray) -> np.ndarray:
            return evaluate(parameters)[0]

        def jacobian(parameters: np.ndarray) -> np.ndarray:
            return evaluate(parameters)[1]

        lower = np.array([-8, -3, -3, -8, -3, -3, -8, -3, -3, -6, -6, -8, -8], dtype=float)
        upper = np.array([4, 3, 3, 4, 3, 3, 4, 3, 3, 1, 1, 1, 1], dtype=float)
        starts = max(1, int(multistart_override if multistart_override is not None else config.scaffold_multistart))
        max_nfev = int(max_nfev_override if max_nfev_override is not None else config.scaffold_max_nfev)
        best = None
        for start_index in range(starts):
            candidate = initial.copy()
            if start_index > 0:
                candidate += rng.normal(0.0, 0.25, size=candidate.size)
                candidate = np.minimum(np.maximum(candidate, lower + 1e-5), upper - 1e-5)
            result = least_squares(residual, candidate, jac=jacobian, bounds=(lower, upper), loss="soft_l1", f_scale=0.70, max_nfev=max_nfev, xtol=1e-7, ftol=1e-7, gtol=1e-7)
            if best is None or result.cost < best.cost:
                best = result
        if best is None or not np.all(np.isfinite(best.x)):
            raise RuntimeError("Demographic scaffold parameter estimation failed")
        active = (np.isclose(best.x, lower, atol=5e-4) | np.isclose(best.x, upper, atol=5e-4)).astype(int)
        return cls(climate, best.x, config.integration_steps, use_climate, float(best.cost), bool(best.success), active, int(best.nfev))

    def climate_scores(self, data: pd.DataFrame) -> np.ndarray:
        return self.climate.transform(data) if self.climate is not None else np.zeros((len(data), 2))

    def predict(self, data: pd.DataFrame, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.integrate_arrays(data, self.climate_scores(data), self.parameters, self.integration_steps if steps is None else steps)

    def parameter_table(self, label: str) -> pd.DataFrame:
        names = ["q_intercept", "q_PC1", "q_PC2", "F_intercept", "F_PC1", "F_PC2", "H_intercept", "H_PC1", "H_PC2", "log_rho", "log_mu", "log_a", "log_b"]
        rows = []
        for index, (name, estimate) in enumerate(zip(names, self.parameters)):
            rows.append({
                "fit": label,
                "parameter": name,
                "estimate_internal": float(estimate),
                "estimate_natural": float(np.exp(estimate)) if index >= 9 else float(estimate),
                "active_bound": int(self.active_bounds[index]),
                "fit_cost": self.fit_cost,
                "fit_success": self.fit_success,
                "nfev": self.nfev,
                "uses_climate": self.use_climate,
            })
        return pd.DataFrame(rows)

def fit_positive_teacher(data: pd.DataFrame, indices: np.ndarray, transform: FeatureTransform, stage: str, spec: ModelSpec, config: Configuration, seed: int) -> ExtraTreesRegressor | None:
    x = transform.transform(data.iloc[indices])
    key = "u" if stage == "young" else "v"
    positive = data.iloc[indices][f"positive_{key}1"].to_numpy(int) == 1 if spec.use_occurrence else np.ones(len(indices), dtype=bool)
    if positive.sum() < 10:
        return None
    y = data.iloc[indices][f"{key}1_log"].to_numpy(float)
    model = ExtraTreesRegressor(
        n_estimators=config.teacher_trees,
        max_depth=config.teacher_max_depth,
        min_samples_leaf=config.teacher_min_leaf,
        max_features=0.85,
        criterion="squared_error",
        random_state=seed,
        n_jobs=-1,
    )
    sample_weight = None
    if spec.use_precision_weights:
        precision = data.iloc[indices]["young_area_precision" if key == "u" else "mature_area_precision"].to_numpy(float)
        sample_weight = np.clip(precision[positive], 0.5, 3.0)
    model.fit(x[positive], y[positive], sample_weight=sample_weight)
    return model

def cross_fitted_teacher_targets(data: pd.DataFrame, spec: ModelSpec, config: Configuration, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not spec.use_teacher:
        return data.u1_log.to_numpy(float), data.v1_log.to_numpy(float), {"teacher_used": False}
    indices = np.arange(len(data))
    groups = data.siteID.astype(str).to_numpy()
    splits = min(config.teacher_folds, np.unique(groups).size)
    if splits < 2:
        return data.u1_log.to_numpy(float), data.v1_log.to_numpy(float), {"teacher_used": False, "reason": "fewer_than_two_sites"}
    prediction_u = np.full(len(data), np.nan, dtype=float)
    prediction_v = np.full(len(data), np.nan, dtype=float)
    for fold, (train, valid) in enumerate(GroupKFold(splits).split(indices, groups=groups), 1):
        transform = FeatureTransform.fit(data.iloc[train], spec, config, seed + fold * 101)
        model_u = fit_positive_teacher(data, train, transform, "young", spec, config, seed + fold * 1009 + 1)
        model_v = fit_positive_teacher(data, train, transform, "mature", spec, config, seed + fold * 1009 + 2)
        xv = transform.transform(data.iloc[valid])
        prediction_u[valid] = np.maximum(model_u.predict(xv), 0.0) if model_u is not None else data.iloc[valid].u0_log.to_numpy(float)
        prediction_v[valid] = np.maximum(model_v.predict(xv), 0.0) if model_v is not None else data.iloc[valid].v0_log.to_numpy(float)
    actual_u = data.u1_log.to_numpy(float)
    actual_v = data.v1_log.to_numpy(float)
    missing_u = ~np.isfinite(prediction_u)
    missing_v = ~np.isfinite(prediction_v)
    prediction_u[missing_u] = data.u0_log.to_numpy(float)[missing_u]
    prediction_v[missing_v] = data.v0_log.to_numpy(float)[missing_v]
    if spec.use_occurrence:
        mask_u = data.positive_u1.to_numpy(int) == 1
        mask_v = data.positive_v1.to_numpy(int) == 1
    else:
        mask_u = np.ones(len(data), dtype=bool)
        mask_v = np.ones(len(data), dtype=bool)
    rmse_u = float(np.sqrt(mean_squared_error(actual_u[mask_u], prediction_u[mask_u]))) if mask_u.any() else float("nan")
    rmse_v = float(np.sqrt(mean_squared_error(actual_v[mask_v], prediction_v[mask_v]))) if mask_v.any() else float("nan")
    return prediction_u, prediction_v, {"teacher_used": True, "teacher_folds": int(splits), "teacher_positive_RMSE_young": rmse_u, "teacher_positive_RMSE_mature": rmse_v}

@dataclass
class BinaryProbabilityModel:
    classifiers: list[ExtraTreesClassifier]
    calibrator: Any
    calibration_method: str
    constant_probability: float | None
    raw_oof_brier: float
    calibrated_oof_brier: float

    @staticmethod
    def raw_probability(model: ExtraTreesClassifier, x: np.ndarray) -> np.ndarray:
        classes = model.classes_
        probability = model.predict_proba(x)
        return probability[:, int(np.flatnonzero(classes == 1)[0])] if 1 in classes else np.zeros(len(x))

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, groups: np.ndarray, config: Configuration, seed: int) -> "BinaryProbabilityModel":
        unique = np.unique(y)
        if unique.size < 2:
            return cls([], None, "constant", float(unique[0]), 0.0, 0.0)
        n_splits = min(config.occurrence_folds, np.unique(groups).size)
        if n_splits < 2:
            model = ExtraTreesClassifier(n_estimators=config.occurrence_trees, max_depth=9, min_samples_leaf=config.occurrence_min_leaf, max_features=0.85, random_state=seed, n_jobs=-1)
            model.fit(x, y)
            raw = cls.raw_probability(model, x)
            score = float(brier_score_loss(y, raw))
            return cls([model], None, "identity", None, score, score)
        oof = np.zeros(len(y), dtype=float)
        models = []
        for fold, (train, valid) in enumerate(GroupKFold(n_splits).split(x, y, groups), 1):
            model = ExtraTreesClassifier(n_estimators=config.occurrence_trees, max_depth=9, min_samples_leaf=config.occurrence_min_leaf, max_features=0.85, random_state=seed + fold, n_jobs=-1)
            model.fit(x[train], y[train])
            oof[valid] = cls.raw_probability(model, x[valid])
            models.append(model)
        raw = np.clip(oof, config.probability_clip, 1 - config.probability_clip)
        if min(np.sum(y == 0), np.sum(y == 1)) >= 25 and np.unique(raw).size >= 12:
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
            method = "isotonic"
            adjusted = calibrator.predict(raw)
        else:
            calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(logit(raw).reshape(-1, 1), y)
            method = "platt"
            adjusted = calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        return cls(models, calibrator, method, None, float(brier_score_loss(y, raw)), float(brier_score_loss(y, np.clip(adjusted, 0, 1))))

    def predict_probability(self, x: np.ndarray, clip: float) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(x), self.constant_probability)
        raw = np.mean([self.raw_probability(model, x) for model in self.classifiers], axis=0)
        raw = np.clip(raw, clip, 1 - clip)
        if self.calibration_method == "isotonic":
            probability = self.calibrator.predict(raw)
        elif self.calibration_method == "platt":
            probability = self.calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        else:
            probability = raw
        return np.clip(probability, clip, 1 - clip)

@dataclass
class OccurrenceLayer:
    transform: FeatureTransform
    young: BinaryProbabilityModel
    mature: BinaryProbabilityModel
    probability_clip: float

    @classmethod
    def fit(cls, data: pd.DataFrame, transform: FeatureTransform, config: Configuration, seed: int) -> "OccurrenceLayer":
        x = transform.transform(data)
        groups = data.siteID.astype(str).to_numpy()
        return cls(
            transform,
            BinaryProbabilityModel.fit(x, data.positive_u1.to_numpy(int), groups, config, seed + 1),
            BinaryProbabilityModel.fit(x, data.positive_v1.to_numpy(int), groups, config, seed + 2),
            config.probability_clip,
        )

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self.transform.transform(data)
        return self.young.predict_probability(x, self.probability_clip), self.mature.predict_probability(x, self.probability_clip)

    def audit(self) -> dict[str, Any]:
        return {
            "young_calibration_method": self.young.calibration_method,
            "mature_calibration_method": self.mature.calibration_method,
            "young_raw_oof_brier": self.young.raw_oof_brier,
            "young_calibrated_oof_brier": self.young.calibrated_oof_brier,
            "mature_raw_oof_brier": self.mature.raw_oof_brier,
            "mature_calibrated_oof_brier": self.mature.calibrated_oof_brier,
        }

@dataclass
class SparseRBFDiscrepancy:
    transform: FeatureTransform
    centers: np.ndarray
    coefficients_young: np.ndarray
    coefficients_mature: np.ndarray
    gammas: tuple[float, ...]
    kernel_weights: tuple[float, ...]
    ridges: tuple[float, float]

    @staticmethod
    def select_centers(x: np.ndarray, count: int, seed: int) -> np.ndarray:
        n = len(x)
        count = min(max(1, int(count)), n)
        if count >= n:
            return x.copy()
        model = MiniBatchKMeans(n_clusters=count, random_state=seed, batch_size=min(1024, max(128, n)), n_init=3, max_iter=100)
        model.fit(x)
        return model.cluster_centers_.copy()

    @staticmethod
    def rbf_features(x: np.ndarray, centers: np.ndarray, gammas: tuple[float, ...], weights: tuple[float, ...]) -> np.ndarray:
        if len(gammas) != len(weights):
            raise ValueError("kernel_gammas and kernel_weights must have equal length")
        distance = cdist(x, centers, metric="sqeuclidean")
        blocks = [math.sqrt(weight) * np.exp(-gamma * distance) for gamma, weight in zip(gammas, weights)]
        return np.column_stack([np.ones(len(x)), *blocks])

    @staticmethod
    def solve_ridge(phi: np.ndarray, target: np.ndarray, sample_weight: np.ndarray, ridge: float, jitter: float) -> np.ndarray:
        solution = SparseRBFDiscrepancy.solve_ridge_multi(phi, np.asarray(target, dtype=float).reshape(-1, 1), sample_weight, ridge, jitter)
        return solution[:, 0]

    @staticmethod
    def solve_ridge_multi(phi: np.ndarray, target: np.ndarray, sample_weight: np.ndarray, ridge: float, jitter: float) -> np.ndarray:
        w = np.sqrt(np.maximum(sample_weight, 1e-12))
        design = phi * w[:, None]
        response = np.asarray(target, dtype=float) * w[:, None]
        penalty = np.eye(phi.shape[1]) * (ridge + jitter)
        penalty[0, 0] = jitter
        matrix = design.T @ design + penalty
        rhs = design.T @ response
        try:
            return cho_solve(cho_factor(matrix, lower=True, check_finite=False), rhs, check_finite=False)
        except (LinAlgError, ValueError):
            return np.linalg.solve(matrix + 1e-7 * np.eye(matrix.shape[0]), rhs)

    @classmethod
    def fit_prepared(
        cls,
        data: pd.DataFrame,
        transform: FeatureTransform,
        centers: np.ndarray,
        target_endpoint_u: np.ndarray,
        target_endpoint_v: np.ndarray,
        scaffold_u: np.ndarray,
        scaffold_v: np.ndarray,
        gate: float,
        spec: ModelSpec,
        config: Configuration,
        ridge_scale: float,
        gammas: tuple[float, ...] | None = None,
        kernel_weights: tuple[float, ...] | None = None,
        sample_weight_young: np.ndarray | None = None,
        sample_weight_mature: np.ndarray | None = None,
    ) -> "SparseRBFDiscrepancy":
        x = transform.transform(data)
        gammas = tuple(config.kernel_gammas if gammas is None else gammas)
        kernel_weights = tuple(config.kernel_weights if kernel_weights is None else kernel_weights)
        phi = cls.rbf_features(x, centers, gammas, kernel_weights)
        base_u = data.u0_log.to_numpy(float)
        base_v = data.v0_log.to_numpy(float)
        residual_u = target_endpoint_u - base_u - gate * (scaffold_u - base_u)
        residual_v = target_endpoint_v - base_v - gate * (scaffold_v - base_v)
        mask_u = data.positive_u1.to_numpy(int) == 1 if spec.use_occurrence else np.ones(len(data), dtype=bool)
        mask_v = data.positive_v1.to_numpy(int) == 1 if spec.use_occurrence else np.ones(len(data), dtype=bool)
        if mask_u.sum() < 5:
            mask_u = np.ones(len(data), dtype=bool)
        if mask_v.sum() < 5:
            mask_v = np.ones(len(data), dtype=bool)
        if sample_weight_young is not None:
            weight_u = np.maximum(np.asarray(sample_weight_young, dtype=float), 1e-12)
        elif spec.use_precision_weights:
            weight_u = np.clip(data.young_area_precision.to_numpy(float), 0.5, 3.0)
        else:
            weight_u = np.ones(len(data))
        if sample_weight_mature is not None:
            weight_v = np.maximum(np.asarray(sample_weight_mature, dtype=float), 1e-12)
        elif spec.use_precision_weights:
            weight_v = np.clip(data.mature_area_precision.to_numpy(float), 0.5, 3.0)
        else:
            weight_v = np.ones(len(data))
        ridge_u = config.young_kernel_ridge * ridge_scale
        ridge_v = config.mature_kernel_ridge * ridge_scale
        coefficients_u = cls.solve_ridge(phi[mask_u], residual_u[mask_u], weight_u[mask_u], ridge_u, config.kernel_jitter)
        coefficients_v = cls.solve_ridge(phi[mask_v], residual_v[mask_v], weight_v[mask_v], ridge_v, config.kernel_jitter)
        return cls(transform, centers, coefficients_u, coefficients_v, gammas, kernel_weights, (ridge_u, ridge_v))

    def features(self, data: pd.DataFrame) -> np.ndarray:
        return self.rbf_features(self.transform.transform(data), self.centers, self.gammas, self.kernel_weights)

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        phi = self.features(data)
        return phi @ self.coefficients_young, phi @ self.coefficients_mature

    def ood_diagnostics(self, data: pd.DataFrame) -> pd.DataFrame:
        x = self.transform.transform(data)
        distance = cdist(x, self.centers, metric="sqeuclidean")
        similarities = np.zeros_like(distance)
        for gamma, weight in zip(self.gammas, self.kernel_weights):
            similarities += weight * np.exp(-gamma * distance)
        probability = similarities / np.maximum(similarities.sum(axis=1, keepdims=True), 1e-12)
        return pd.DataFrame({
            "nearest_feature_distance": np.sqrt(distance.min(axis=1)),
            "max_kernel_similarity": similarities.max(axis=1),
            "effective_kernel_neighbors": 1.0 / np.maximum(np.sum(probability * probability, axis=1), 1e-12),
        })

    def derivatives(self, data: pd.DataFrame, raw_scale: bool = True) -> dict[str, np.ndarray]:
        x = self.transform.transform(data)
        n, p = x.shape
        gradient_u = np.zeros((n, p), dtype=float)
        gradient_v = np.zeros((n, p), dtype=float)
        hdiag_u = np.zeros((n, p), dtype=float)
        hdiag_v = np.zeros((n, p), dtype=float)
        offset = 1
        distance = cdist(x, self.centers, metric="sqeuclidean")
        for gamma, weight in zip(self.gammas, self.kernel_weights):
            block = math.sqrt(weight) * np.exp(-gamma * distance)
            beta_u = self.coefficients_young[offset: offset + len(self.centers)]
            beta_v = self.coefficients_mature[offset: offset + len(self.centers)]
            for feature in range(p):
                delta = x[:, feature, None] - self.centers[None, :, feature]
                first = -2.0 * gamma * delta * block
                second = (4.0 * gamma * gamma * delta * delta - 2.0 * gamma) * block
                gradient_u[:, feature] += first @ beta_u
                gradient_v[:, feature] += first @ beta_v
                hdiag_u[:, feature] += second @ beta_u
                hdiag_v[:, feature] += second @ beta_v
            offset += len(self.centers)
        if raw_scale:
            scale = self.transform.derivative_scale_matrix(data)
            gradient_u = gradient_u * scale
            gradient_v = gradient_v * scale
            hdiag_u = hdiag_u * scale * scale
            hdiag_v = hdiag_v * scale * scale
        return {"gradient_u": gradient_u, "gradient_v": gradient_v, "hessian_diag_u": hdiag_u, "hessian_diag_v": hdiag_v}

    def mean_abs_hessian_pairs(self, data: pd.DataFrame, raw_scale: bool = True) -> pd.DataFrame:
        x = self.transform.transform(data)
        p = x.shape[1]
        scale = self.transform.derivative_scale_matrix(data) if raw_scale else np.ones((len(data), p))
        distance = cdist(x, self.centers, metric="sqeuclidean")
        rows = []
        for left in range(p):
            for right in range(left + 1, p):
                hu = np.zeros(len(x), dtype=float)
                hv = np.zeros(len(x), dtype=float)
                offset = 1
                dl = x[:, left, None] - self.centers[None, :, left]
                dr = x[:, right, None] - self.centers[None, :, right]
                for gamma, weight in zip(self.gammas, self.kernel_weights):
                    block = math.sqrt(weight) * np.exp(-gamma * distance)
                    mixed = 4.0 * gamma * gamma * dl * dr * block
                    beta_u = self.coefficients_young[offset: offset + len(self.centers)]
                    beta_v = self.coefficients_mature[offset: offset + len(self.centers)]
                    hu += mixed @ beta_u
                    hv += mixed @ beta_v
                    offset += len(self.centers)
                hu *= scale[:, left] * scale[:, right]
                hv *= scale[:, left] * scale[:, right]
                rows.append({"feature_1": self.transform.feature_names[left], "feature_2": self.transform.feature_names[right], "mean_abs_hessian_young": float(np.mean(np.abs(hu))), "mean_abs_hessian_mature": float(np.mean(np.abs(hv))), "mean_hessian_young": float(np.mean(hu)), "mean_hessian_mature": float(np.mean(hv))})
        return pd.DataFrame(rows)

    def mathematical_audit(self) -> dict[str, float]:
        rows = {}
        for stage, coefficients in [("young", self.coefficients_young), ("mature", self.coefficients_mature)]:
            offset = 1
            bound = 0.0
            l1 = 0.0
            for gamma, weight in zip(self.gammas, self.kernel_weights):
                beta = coefficients[offset: offset + len(self.centers)]
                l1 += float(np.linalg.norm(beta, 1))
                bound += float(np.linalg.norm(beta, 1)) * math.sqrt(weight) * math.sqrt(2.0 * gamma / math.e)
                offset += len(self.centers)
            rows[f"rbf_coefficient_L1_{stage}"] = l1
            rows[f"finite_Lipschitz_upper_weighted_space_{stage}"] = bound
        return rows

@dataclass
class TRACETrainingContext:
    data: pd.DataFrame
    spec: ModelSpec
    config: Configuration
    seed: int
    scaffold: DemographicScaffold | None
    transform: FeatureTransform
    centers: np.ndarray
    occurrence: OccurrenceLayer | None
    teacher_u: np.ndarray
    teacher_v: np.ndarray
    teacher_audit: dict[str, Any]
    scaffold_u: np.ndarray
    scaffold_v: np.ndarray
    estimation_weight_young: np.ndarray | None
    estimation_weight_mature: np.ndarray | None
    estimation_weight_audit: pd.DataFrame

    @classmethod
    def prepare(cls, data: pd.DataFrame, spec: ModelSpec, config: Configuration, seed: int) -> "TRACETrainingContext":
        if spec.use_variance_weights:
            weight_u, weight_v, weight_audit = cross_fitted_estimation_weights(data, config, seed + 17)
        else:
            weight_u, weight_v = None, None
            weight_audit = pd.DataFrame([{"mode": "disabled", "rows": len(data), "sites": int(data.siteID.nunique())}])
        scaffold = DemographicScaffold.fit(
            data,
            config,
            seed + 101,
            spec.use_climate,
            spec.use_precision_weights,
            sample_weight_young=weight_u,
            sample_weight_mature=weight_v,
        ) if spec.use_scaffold else None
        teacher_u, teacher_v, teacher_audit = cross_fitted_teacher_targets(data, spec, config, seed + 202)
        transform = FeatureTransform.fit(data, spec, config, seed + 303)
        predictor_availability_audit(data, spec, transform.feature_names)
        x = transform.transform(data)
        centers = SparseRBFDiscrepancy.select_centers(x, config.rbf_centers, seed + 404)
        occurrence = OccurrenceLayer.fit(data, transform, config, seed + 505) if spec.use_occurrence else None
        base_u = data.u0_log.to_numpy(float)
        base_v = data.v0_log.to_numpy(float)
        scaffold_u, scaffold_v = scaffold.predict(data) if scaffold is not None else (base_u.copy(), base_v.copy())
        if occurrence is not None:
            teacher_audit = {**teacher_audit, **occurrence.audit()}
        teacher_audit = {**teacher_audit, "variance_weighting": bool(spec.use_variance_weights), "legacy_precision_weighting": bool(spec.use_precision_weights)}
        return cls(data, spec, config, seed, scaffold, transform, centers, occurrence, teacher_u, teacher_v, teacher_audit, scaffold_u, scaffold_v, weight_u, weight_v, weight_audit)

    def endpoint_targets(self, hyper: HyperParameters) -> tuple[np.ndarray, np.ndarray]:
        actual_u = self.data.u1_log.to_numpy(float)
        actual_v = self.data.v1_log.to_numpy(float)
        weight = float(np.clip(hyper.distillation_weight, 0.0, 1.0)) if self.spec.use_teacher else 1.0
        return (1.0 - weight) * self.teacher_u + weight * actual_u, (1.0 - weight) * self.teacher_v + weight * actual_v

    def fit(self, hyper: HyperParameters) -> "TRACEModel":
        gate = float(np.clip(hyper.gate, 0.0, 1.0)) if self.spec.use_scaffold else 0.0
        distillation_weight = float(np.clip(hyper.distillation_weight, 0.0, 1.0)) if self.spec.use_teacher else 1.0
        occurrence_gate = float(np.clip(hyper.occurrence_gate, 0.0, 1.0)) if self.spec.use_occurrence else 0.0
        gamma_scale = max(float(hyper.gamma_scale), 1e-6)
        kernel_alpha = float(np.clip(hyper.kernel_alpha, 0.0, 1.0))
        effective = HyperParameters(gate, float(hyper.ridge_scale), distillation_weight, occurrence_gate, gamma_scale, kernel_alpha)
        target_u, target_v = self.endpoint_targets(effective)
        gammas = tuple(float(gamma * gamma_scale) for gamma in self.config.kernel_gammas)
        if len(gammas) == 2:
            kernel_weights = (kernel_alpha, 1.0 - kernel_alpha)
        else:
            raw = np.asarray(self.config.kernel_weights, dtype=float)
            raw = np.maximum(raw, 0.0)
            raw = raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1.0 / len(raw))
            kernel_weights = tuple(float(value) for value in raw)
        discrepancy = SparseRBFDiscrepancy.fit_prepared(
            self.data,
            self.transform,
            self.centers,
            target_u,
            target_v,
            self.scaffold_u,
            self.scaffold_v,
            gate,
            self.spec,
            self.config,
            effective.ridge_scale,
            gammas=gammas,
            kernel_weights=kernel_weights,
            sample_weight_young=self.estimation_weight_young,
            sample_weight_mature=self.estimation_weight_mature,
        )
        return TRACEModel(self.spec, effective, self.scaffold, discrepancy, self.occurrence, self.teacher_audit, self.estimation_weight_audit.copy())

@dataclass
class TRACEModel:
    spec: ModelSpec
    hyper: HyperParameters
    scaffold: DemographicScaffold | None
    discrepancy: SparseRBFDiscrepancy
    occurrence: OccurrenceLayer | None
    teacher_audit: dict[str, Any]
    estimation_weight_audit: pd.DataFrame

    @classmethod
    def fit(cls, data: pd.DataFrame, spec: ModelSpec, hyper: HyperParameters, config: Configuration, seed: int) -> "TRACEModel":
        return TRACETrainingContext.prepare(data, spec, config, seed).fit(hyper)

    def components(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        base_u = data.u0_log.to_numpy(float)
        base_v = data.v0_log.to_numpy(float)
        scaffold_u, scaffold_v = self.scaffold.predict(data) if self.scaffold is not None else (base_u.copy(), base_v.copy())
        residual_u, residual_v = self.discrepancy.predict(data)
        gated_u = self.hyper.gate * (scaffold_u - base_u)
        gated_v = self.hyper.gate * (scaffold_v - base_v)
        magnitude_u = np.maximum(base_u + gated_u + residual_u, 0.0)
        magnitude_v = np.maximum(base_v + gated_v + residual_v, 0.0)
        probability_u, probability_v = self.occurrence.predict(data) if self.occurrence is not None else (np.ones(len(data)), np.ones(len(data)))
        original_mean_u = probability_u * np.expm1(magnitude_u)
        original_mean_v = probability_v * np.expm1(magnitude_v)
        hurdle_original_log_u = np.log1p(np.maximum(original_mean_u, 0.0))
        hurdle_original_log_v = np.log1p(np.maximum(original_mean_v, 0.0))
        legacy_log_expectation_u = probability_u * magnitude_u
        legacy_log_expectation_v = probability_v * magnitude_v
        h = self.hyper.occurrence_gate if self.occurrence is not None else 0.0
        prediction_u = (1.0 - h) * magnitude_u + h * hurdle_original_log_u
        prediction_v = (1.0 - h) * magnitude_v + h * hurdle_original_log_v
        point_scale_u = np.expm1(np.maximum(prediction_u, 0.0))
        point_scale_v = np.expm1(np.maximum(prediction_v, 0.0))
        return {
            "base_u": base_u,
            "base_v": base_v,
            "scaffold_u": scaffold_u,
            "scaffold_v": scaffold_v,
            "scaffold_delta_u": scaffold_u - base_u,
            "scaffold_delta_v": scaffold_v - base_v,
            "gated_scaffold_u": gated_u,
            "gated_scaffold_v": gated_v,
            "discrepancy_u": residual_u,
            "discrepancy_v": residual_v,
            "magnitude_u": magnitude_u,
            "magnitude_v": magnitude_v,
            "positive_probability_u": probability_u,
            "positive_probability_v": probability_v,
            "hurdle_original_log_u": hurdle_original_log_u,
            "hurdle_original_log_v": hurdle_original_log_v,
            "legacy_log_expectation_u": legacy_log_expectation_u,
            "legacy_log_expectation_v": legacy_log_expectation_v,
            "distribution_log_mean_u": legacy_log_expectation_u,
            "distribution_log_mean_v": legacy_log_expectation_v,
            "pred_u": prediction_u,
            "pred_v": prediction_v,
            "original_scale_mean_u": original_mean_u,
            "original_scale_mean_v": original_mean_v,
            "point_scale_u": point_scale_u,
            "point_scale_v": point_scale_v,
        }

    def point_prediction_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        c = self.components(data)
        ood = self.discrepancy.ood_diagnostics(data)
        frame = pd.DataFrame({
            "pred_u_log": c["pred_u"],
            "pred_v_log": c["pred_v"],
            "scaffold_u_log": c["scaffold_u"],
            "scaffold_v_log": c["scaffold_v"],
            "scaffold_delta_u_log": c["scaffold_delta_u"],
            "scaffold_delta_v_log": c["scaffold_delta_v"],
            "gated_scaffold_u_log": c["gated_scaffold_u"],
            "gated_scaffold_v_log": c["gated_scaffold_v"],
            "discrepancy_u_log": c["discrepancy_u"],
            "discrepancy_v_log": c["discrepancy_v"],
            "positive_magnitude_u_log": c["magnitude_u"],
            "positive_magnitude_v_log": c["magnitude_v"],
            "positive_probability_u": c["positive_probability_u"],
            "positive_probability_v": c["positive_probability_v"],
            "hurdle_original_mean_u_log": c["hurdle_original_log_u"],
            "hurdle_original_mean_v_log": c["hurdle_original_log_v"],
            "legacy_hurdle_expectation_u_log": c["legacy_log_expectation_u"],
            "legacy_hurdle_expectation_v_log": c["legacy_log_expectation_v"],
            "occurrence_adjustment_u_log": c["pred_u"] - c["magnitude_u"],
            "occurrence_adjustment_v_log": c["pred_v"] - c["magnitude_v"],
            "point_estimate_u_kha": c["point_scale_u"],
            "point_estimate_v_kha": c["point_scale_v"],
            "point_mean_u_kha": c["original_scale_mean_u"],
            "point_mean_v_kha": c["original_scale_mean_v"],
        })
        return pd.concat([frame, ood.reset_index(drop=True)], axis=1)

@dataclass
class DirectBenchmark:
    name: str
    transform: FeatureTransform
    young_model: Any
    mature_model: Any
    occurrence: OccurrenceLayer | None
    parameters: dict[str, Any]

    @classmethod
    def fit(cls, name: str, data: pd.DataFrame, spec: ModelSpec, config: Configuration, seed: int, parameters: dict[str, Any] | None = None) -> "DirectBenchmark":
        parameters = dict(parameters or {})
        transform = FeatureTransform.fit(data, spec, config, seed)
        x = transform.transform(data)
        y_u = data.u1_log.to_numpy(float)
        y_v = data.v1_log.to_numpy(float)
        hurdle = name == "Hurdle Extra Trees"
        mask_u = data.positive_u1.to_numpy(int) == 1 if hurdle else np.ones(len(data), dtype=bool)
        mask_v = data.positive_v1.to_numpy(int) == 1 if hurdle else np.ones(len(data), dtype=bool)
        if name in {"Extra Trees direct", "Hurdle Extra Trees"}:
            depth = parameters.get("max_depth", 10)
            leaf = parameters.get("min_samples_leaf", 6)
            young_model = ExtraTreesRegressor(n_estimators=max(30, config.teacher_trees), max_depth=depth, min_samples_leaf=leaf, max_features=0.85, random_state=seed + 1, n_jobs=-1)
            mature_model = ExtraTreesRegressor(n_estimators=max(30, config.teacher_trees), max_depth=parameters.get("mature_max_depth", max(5, depth - 2) if depth is not None else None), min_samples_leaf=parameters.get("mature_min_samples_leaf", leaf + 2), max_features=0.85, random_state=seed + 2, n_jobs=-1)
        elif name == "Histogram boosting direct":
            max_leaf_nodes = parameters.get("max_leaf_nodes", 15)
            l2 = parameters.get("l2_regularization", 1.0)
            learning_rate = parameters.get("learning_rate", 0.05)
            young_model = HistGradientBoostingRegressor(max_iter=180, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=15, l2_regularization=l2, random_state=seed + 1)
            mature_model = clone(young_model).set_params(random_state=seed + 2)
        elif name == "Ridge direct":
            alpha = parameters.get("alpha", 10.0)
            young_model = Ridge(alpha=alpha)
            mature_model = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Unknown benchmark: {name}")
        if spec.use_variance_weights:
            weight_u, weight_v, _ = cross_fitted_estimation_weights(data, config, seed + 901)
        elif spec.use_precision_weights:
            weight_u = np.clip(data.young_area_precision.to_numpy(float), 0.5, 3.0)
            weight_v = np.clip(data.mature_area_precision.to_numpy(float), 0.5, 3.0)
        else:
            weight_u = None
            weight_v = None
        young_model.fit(x[mask_u], y_u[mask_u], sample_weight=weight_u[mask_u] if weight_u is not None else None)
        mature_model.fit(x[mask_v], y_v[mask_v], sample_weight=weight_v[mask_v] if weight_v is not None else None)
        occurrence = OccurrenceLayer.fit(data, transform, config, seed + 1000) if hurdle else None
        return cls(name, transform, young_model, mature_model, occurrence, parameters)

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self.transform.transform(data)
        magnitude_u = np.maximum(self.young_model.predict(x), 0.0)
        magnitude_v = np.maximum(self.mature_model.predict(x), 0.0)
        if self.occurrence is None:
            return magnitude_u, magnitude_v
        p_u, p_v = self.occurrence.predict(data)
        h = float(np.clip(self.parameters.get("occurrence_gate", 1.0), 0.0, 1.0))
        a_u = np.log1p(p_u * np.expm1(magnitude_u))
        a_v = np.log1p(p_v * np.expm1(magnitude_v))
        return (1.0 - h) * magnitude_u + h * a_u, (1.0 - h) * magnitude_v + h * a_v

def benchmark_parameter_grid(name: str) -> list[dict[str, Any]]:
    if name == "Extra Trees direct":
        return [
            {"max_depth": 7, "min_samples_leaf": 6},
            {"max_depth": 10, "min_samples_leaf": 6},
            {"max_depth": 10, "min_samples_leaf": 10},
        ]
    if name == "Hurdle Extra Trees":
        structures = [
            {"max_depth": 7, "min_samples_leaf": 6},
            {"max_depth": 10, "min_samples_leaf": 6},
            {"max_depth": 10, "min_samples_leaf": 10},
        ]
        return [{**structure, "occurrence_gate": float(h)} for structure in structures for h in (0.0, 0.5, 1.0)]
    if name == "Histogram boosting direct":
        return [
            {"max_leaf_nodes": 7, "l2_regularization": 1.0, "learning_rate": 0.05},
            {"max_leaf_nodes": 15, "l2_regularization": 1.0, "learning_rate": 0.05},
            {"max_leaf_nodes": 15, "l2_regularization": 2.0, "learning_rate": 0.04},
        ]
    if name == "Ridge direct":
        return [{"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0}]
    raise ValueError(name)

def normalized_pair_rmse(data: pd.DataFrame, pred_u: np.ndarray, pred_v: np.ndarray) -> float:
    rmse_u = np.sqrt(mean_squared_error(data.u1_log, pred_u))
    rmse_v = np.sqrt(mean_squared_error(data.v1_log, pred_v))
    scale_u = np.std(data.u1_log.to_numpy(float)) + 1e-8
    scale_v = np.std(data.v1_log.to_numpy(float)) + 1e-8
    return float(0.5 * (rmse_u / scale_u + rmse_v / scale_v))

def tuning_score(data: pd.DataFrame, pred_u: np.ndarray, pred_v: np.ndarray, zero_weight: float) -> tuple[float, float, float]:
    overall = normalized_pair_rmse(data, pred_u, pred_v)
    parts = []
    for key, pred in [("u", pred_u), ("v", pred_v)]:
        mask = data[f"zero_{key}0"].to_numpy(int) == 1
        if mask.sum() >= 5:
            scale = np.std(data[f"{key}1_log"].to_numpy(float)) + 1e-8
            parts.append(np.sqrt(mean_squared_error(data.loc[mask, f"{key}1_log"], pred[mask])) / scale)
    zero = float(np.mean(parts)) if parts else overall
    return (1.0 - zero_weight) * overall + zero_weight * zero, overall, zero

def default_hyperparameters(spec: ModelSpec) -> HyperParameters:
    return HyperParameters(
        gate=0.25 if spec.use_scaffold else 0.0,
        ridge_scale=1.0,
        distillation_weight=0.75 if spec.use_teacher else 1.0,
        occurrence_gate=1.0 if spec.use_occurrence else 0.0,
        gamma_scale=1.0,
        kernel_alpha=0.50,
    )

def trace_stage_candidates(stage: str, current: HyperParameters, spec: ModelSpec, config: Configuration) -> list[HyperParameters]:
    candidates = []
    if stage in {"representation", "representation_refinement"}:
        for gamma_scale in config.gamma_scale_grid:
            for kernel_alpha in config.kernel_alpha_grid:
                candidates.append(replace(current, gamma_scale=float(gamma_scale), kernel_alpha=float(kernel_alpha)))
    elif stage == "teacher":
        values = config.distillation_weight_grid if spec.use_teacher else (1.0,)
        candidates = [replace(current, distillation_weight=float(value)) for value in values]
    elif stage == "occurrence":
        values = config.occurrence_gate_grid if spec.use_occurrence else (0.0,)
        candidates = [replace(current, occurrence_gate=float(value)) for value in values]
    elif stage == "structural":
        gates = config.gate_grid if spec.use_scaffold else (0.0,)
        for ridge_scale in config.ridge_scale_grid:
            for gate in gates:
                candidates.append(replace(current, gate=float(gate), ridge_scale=float(ridge_scale)))
    else:
        raise ValueError(f"Unknown TRACE tuning stage: {stage}")
    unique = []
    seen = set()
    for candidate in candidates:
        key = tuple(asdict(candidate).values())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique

def hyperparameter_record(hyper: HyperParameters) -> dict[str, float]:
    return {
        "gate": float(hyper.gate),
        "ridge_scale": float(hyper.ridge_scale),
        "distillation_weight": float(hyper.distillation_weight),
        "occurrence_gate": float(hyper.occurrence_gate),
        "gamma_scale": float(hyper.gamma_scale),
        "kernel_alpha": float(hyper.kernel_alpha),
    }

def tune_trace_model(data: pd.DataFrame, spec: ModelSpec, config: Configuration, seed: int) -> tuple[HyperParameters, pd.DataFrame]:
    current = default_hyperparameters(spec)
    if not config.tune_model:
        return current, pd.DataFrame()
    groups = data.siteID.astype(str).to_numpy()
    splits = min(config.inner_folds, np.unique(groups).size)
    if splits < 2:
        return current, pd.DataFrame()
    indices = np.arange(len(data))
    prepared = []
    for inner_fold, (train_index, valid_index) in enumerate(GroupKFold(splits).split(indices, groups=groups), 1):
        train = data.iloc[train_index].reset_index(drop=True)
        valid = data.iloc[valid_index].reset_index(drop=True)
        if set(train.siteID.astype(str)) & set(valid.siteID.astype(str)):
            raise RuntimeError("Inner-CV site overlap detected")
        context = TRACETrainingContext.prepare(train, spec, config, seed + inner_fold * 100003)
        prepared.append((inner_fold, context, valid))
    stages = ["representation", "teacher", "occurrence", "structural"]
    if config.tuning_refinement and len(config.gamma_scale_grid) * len(config.kernel_alpha_grid) > 1:
        stages.append("representation_refinement")
    all_rows = []
    for stage_index, stage in enumerate(stages, 1):
        candidates = trace_stage_candidates(stage, current, spec, config)
        rows = []
        for candidate_index, candidate in enumerate(candidates):
            for inner_fold, context, valid in prepared:
                model = context.fit(candidate)
                components = model.components(valid)
                objective, overall, zero = tuning_score(valid, components["pred_u"], components["pred_v"], config.tuning_zero_weight)
                rows.append({"model": spec.name, "stage": stage, "stage_index": stage_index, "candidate_index": candidate_index, "inner_fold": inner_fold, **hyperparameter_record(candidate), "objective": objective, "overall_normalized_RMSE": overall, "zero_origin_normalized_RMSE": zero, "candidate_refitted": True, "site_grouped_inner_cv": True})
        table = pd.DataFrame(rows)
        hyper_columns = ["gate", "ridge_scale", "distillation_weight", "occurrence_gate", "gamma_scale", "kernel_alpha"]
        summary = table.groupby(["model", "stage", "stage_index", "candidate_index", *hyper_columns], as_index=False).agg(objective=("objective", "mean"), objective_sd=("objective", "std"), overall_normalized_RMSE=("overall_normalized_RMSE", "mean"), zero_origin_normalized_RMSE=("zero_origin_normalized_RMSE", "mean"))
        best = summary.sort_values(["objective", "candidate_index"]).iloc[0]
        current = HyperParameters(float(best.gate), float(best.ridge_scale), float(best.distillation_weight), float(best.occurrence_gate), float(best.gamma_scale), float(best.kernel_alpha))
        selected_mask = np.ones(len(table), dtype=bool)
        for column, value in hyperparameter_record(current).items():
            selected_mask &= np.isclose(table[column].to_numpy(float), value)
        table["selected"] = selected_mask
        summary["inner_fold"] = 0
        selected_summary = np.ones(len(summary), dtype=bool)
        for column, value in hyperparameter_record(current).items():
            selected_summary &= np.isclose(summary[column].to_numpy(float), value)
        summary["selected"] = selected_summary
        summary["candidate_refitted"] = True
        summary["site_grouped_inner_cv"] = True
        all_rows.extend([table, summary])
    return current, pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()

def tune_direct_benchmark(data: pd.DataFrame, name: str, spec: ModelSpec, config: Configuration, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    grid = benchmark_parameter_grid(name)
    if not config.tune_benchmarks:
        return grid[min(1, len(grid) - 1)], pd.DataFrame()
    groups = data.siteID.astype(str).to_numpy()
    splits = min(config.inner_folds, np.unique(groups).size)
    if splits < 2:
        return grid[0], pd.DataFrame()
    rows = []
    indices = np.arange(len(data))
    for inner_fold, (train_index, valid_index) in enumerate(GroupKFold(splits).split(indices, groups=groups), 1):
        train = data.iloc[train_index].reset_index(drop=True)
        valid = data.iloc[valid_index].reset_index(drop=True)
        for candidate_index, parameters in enumerate(grid):
            model = DirectBenchmark.fit(name, train, spec, config, seed + inner_fold * 10007 + candidate_index * 503, parameters)
            pred_u, pred_v = model.predict(valid)
            objective, overall, zero = tuning_score(valid, pred_u, pred_v, config.tuning_zero_weight)
            rows.append({"model": name, "inner_fold": inner_fold, "candidate": candidate_index, "parameters": json.dumps(parameters, sort_keys=True), "objective": objective, "overall_normalized_RMSE": overall, "zero_origin_normalized_RMSE": zero})
    table = pd.DataFrame(rows)
    summary = table.groupby(["model", "candidate", "parameters"], as_index=False).agg(objective=("objective", "mean"), objective_sd=("objective", "std"), overall_normalized_RMSE=("overall_normalized_RMSE", "mean"), zero_origin_normalized_RMSE=("zero_origin_normalized_RMSE", "mean"))
    best = summary.sort_values(["objective", "candidate"]).iloc[0]
    candidate = int(best.candidate)
    table["selected"] = table.candidate == candidate
    summary["inner_fold"] = 0
    summary["selected"] = summary.candidate == candidate
    return grid[candidate], pd.concat([table, summary], ignore_index=True, sort=False)

def forecast_sampling_area(data: pd.DataFrame, key: str) -> np.ndarray:
    stage = "young" if key == "u" else "mature"
    reference = 200.0 if key == "u" else 800.0
    column = f"area_{stage}_t0_m2_reconstructed"
    if column not in data:
        return np.full(len(data), reference, dtype=float)
    area = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(area) & (area > 0), area, reference)

def expected_count_from_components(data: pd.DataFrame, components: dict[str, np.ndarray], key: str) -> np.ndarray:
    area = forecast_sampling_area(data, key)
    density_mean = components[f"positive_probability_{key}"] * np.expm1(np.maximum(components[f"magnitude_{key}"], 0.0))
    return np.maximum(density_mean * area / 10.0, 0.0)

def persistence_reference_components(data: pd.DataFrame) -> dict[str, np.ndarray]:
    base_u = data.u0_log.to_numpy(float)
    base_v = data.v0_log.to_numpy(float)
    ones = np.ones(len(data), dtype=float)
    return {
        "pred_u": base_u.copy(),
        "pred_v": base_v.copy(),
        "magnitude_u": base_u.copy(),
        "magnitude_v": base_v.copy(),
        "positive_probability_u": ones.copy(),
        "positive_probability_v": ones.copy(),
        "distribution_log_mean_u": base_u.copy(),
        "distribution_log_mean_v": base_v.copy(),
    }

def predictive_variance_from_components(data: pd.DataFrame, components: dict[str, np.ndarray], stochastic: dict[str, float], config: Configuration) -> tuple[np.ndarray, np.ndarray]:
    dt = data.dt_years.to_numpy(float)
    expected_u = expected_count_from_components(data, components, "u")
    expected_v = expected_count_from_components(data, components, "v")
    offset = max(float(config.variance_count_offset), 1e-6)
    variance_u = stochastic["sigma_young"] ** 2 * dt + stochastic["observation_scale_young"] ** 2 / (expected_u + offset)
    variance_v = stochastic["sigma_mature"] ** 2 * dt + stochastic["observation_scale_mature"] ** 2 / (expected_v + offset)
    return np.maximum(variance_u, 1e-10), np.maximum(variance_v, 1e-10)

def normalize_inverse_variance_weights(values: np.ndarray, config: Configuration) -> tuple[np.ndarray, dict[str, float]]:
    weight = np.asarray(values, dtype=float)
    finite = weight[np.isfinite(weight) & (weight > 0)]
    if finite.size == 0:
        return np.ones(len(weight), dtype=float), {"raw_min": np.nan, "raw_max": np.nan, "clip_low": np.nan, "clip_high": np.nan, "normalized_min": 1.0, "normalized_max": 1.0}
    low = float(np.quantile(finite, config.variance_weight_lower_quantile))
    high = float(np.quantile(finite, config.variance_weight_upper_quantile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= 0:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high < low:
        low, high = high, low
    clipped = np.clip(np.where(np.isfinite(weight) & (weight > 0), weight, np.median(finite)), max(low, 1e-12), max(high, max(low, 1e-12)))
    mean = float(np.mean(clipped))
    normalized = clipped / mean if mean > 0 else np.ones(len(clipped), dtype=float)
    return normalized, {"raw_min": float(np.min(finite)), "raw_max": float(np.max(finite)), "clip_low": low, "clip_high": high, "normalized_min": float(np.min(normalized)), "normalized_max": float(np.max(normalized))}

def cross_fitted_estimation_weights(data: pd.DataFrame, config: Configuration, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    groups = data.siteID.astype(str).to_numpy()
    unique_sites = np.unique(groups)
    splits = min(max(2, int(config.variance_weight_folds)), unique_sites.size)
    if splits < 2:
        return np.ones(len(data)), np.ones(len(data)), pd.DataFrame([{"mode": "uniform_fallback", "rows": len(data), "sites": int(unique_sites.size)}])
    indices = np.arange(len(data))
    raw_u = np.full(len(data), np.nan, dtype=float)
    raw_v = np.full(len(data), np.nan, dtype=float)
    rows = []
    for fold, (train_index, valid_index) in enumerate(GroupKFold(splits).split(indices, groups=groups), 1):
        train = data.iloc[train_index].reset_index(drop=True)
        valid = data.iloc[valid_index].reset_index(drop=True)
        if set(train.siteID.astype(str)) & set(valid.siteID.astype(str)):
            raise RuntimeError("Variance-weight cross-fitting site overlap detected")
        train_components = persistence_reference_components(train)
        stochastic = estimate_stochastic_layer(train, train_components, config)
        valid_components = persistence_reference_components(valid)
        variance_u, variance_v = predictive_variance_from_components(valid, valid_components, stochastic, config)
        raw_u[valid_index] = 1.0 / variance_u
        raw_v[valid_index] = 1.0 / variance_v
        rows.append({"fold": fold, "train_sites": int(train.siteID.nunique()), "validation_sites": int(valid.siteID.nunique()), "site_overlap": 0, "reference": "persistence", "sigma_young": stochastic["sigma_young"], "sigma_mature": stochastic["sigma_mature"], "observation_scale_young": stochastic["observation_scale_young"], "observation_scale_mature": stochastic["observation_scale_mature"]})
    if np.any(~np.isfinite(raw_u)) or np.any(~np.isfinite(raw_v)):
        raise RuntimeError("Cross-fitted inverse-variance weights contain non-finite values")
    weight_u, audit_u = normalize_inverse_variance_weights(raw_u, config)
    weight_v, audit_v = normalize_inverse_variance_weights(raw_v, config)
    summary = {"fold": 0, "train_sites": int(data.siteID.nunique()), "validation_sites": int(data.siteID.nunique()), "site_overlap": 0, "reference": "persistence_cross_fitted", **{f"young_{k}": v for k, v in audit_u.items()}, **{f"mature_{k}": v for k, v in audit_v.items()}, "young_mean": float(np.mean(weight_u)), "mature_mean": float(np.mean(weight_v))}
    rows.append(summary)
    return weight_u, weight_v, pd.DataFrame(rows)

def estimate_stochastic_layer(data: pd.DataFrame, components: dict[str, np.ndarray], config: Configuration) -> dict[str, float]:
    center_u = components.get("distribution_log_mean_u", components["pred_u"])
    center_v = components.get("distribution_log_mean_v", components["pred_v"])
    residual_u = data.u1_log.to_numpy(float) - center_u
    residual_v = data.v1_log.to_numpy(float) - center_v
    dt = data.dt_years.to_numpy(float)
    expected_u = expected_count_from_components(data, components, "u")
    expected_v = expected_count_from_components(data, components, "v")
    offset = max(float(config.variance_count_offset), 1e-6)
    obs_u = 1.0 / (expected_u + offset)
    obs_v = 1.0 / (expected_v + offset)

    def estimate(residual: np.ndarray, observation_variance: np.ndarray) -> tuple[float, float]:
        squared = residual * residual
        threshold = np.quantile(squared[np.isfinite(squared)], config.process_trim_quantile)
        keep = np.isfinite(squared) & (squared <= threshold)
        design = np.column_stack([dt[keep], observation_variance[keep]])
        coefficients, _ = nnls(design, squared[keep])
        process_sigma = float(np.clip(math.sqrt(max(coefficients[0], 1e-10)), 0.002, 2.0))
        observation_scale = float(np.clip(math.sqrt(max(coefficients[1], 1e-10)), 0.005, 5.0))
        return process_sigma, observation_scale

    sigma_u, observation_u = estimate(residual_u, obs_u)
    sigma_v, observation_v = estimate(residual_v, obs_v)
    standard_u = residual_u / np.sqrt(np.maximum(sigma_u * sigma_u * dt + observation_u * observation_u * obs_u, 1e-10))
    standard_v = residual_v / np.sqrt(np.maximum(sigma_v * sigma_v * dt + observation_v * observation_v * obs_v, 1e-10))
    correlation = np.corrcoef(np.clip(standard_u, -4, 4), np.clip(standard_v, -4, 4))[0, 1]
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "sigma_young": sigma_u,
        "sigma_mature": sigma_v,
        "observation_scale_young": observation_u,
        "observation_scale_mature": observation_v,
        "endpoint_noise_correlation": float(np.clip(correlation, -0.90, 0.90)),
        "count_offset": offset,
    }

def fit_stochastic_layer_cross_fitted(data: pd.DataFrame, spec: ModelSpec, hyper: HyperParameters, config: Configuration, seed: int) -> tuple[dict[str, float], pd.DataFrame]:
    groups = data.siteID.astype(str).to_numpy()
    splits = min(config.stochastic_folds, np.unique(groups).size)
    fields = ["pred_u", "pred_v", "magnitude_u", "magnitude_v", "positive_probability_u", "positive_probability_v", "distribution_log_mean_u", "distribution_log_mean_v"]
    assembled = {field: np.full(len(data), np.nan, dtype=float) for field in fields}
    rows = []
    if splits < 2:
        model = TRACEModel.fit(data, spec, hyper, config, seed)
        components = model.components(data)
        stochastic = estimate_stochastic_layer(data, components, config)
        rows.append({"fold": 0, "train_sites": int(data.siteID.nunique()), "validation_sites": int(data.siteID.nunique()), "cross_fitted": False})
        return stochastic, pd.DataFrame(rows)
    indices = np.arange(len(data))
    for fold, (train_index, valid_index) in enumerate(GroupKFold(splits).split(indices, groups=groups), 1):
        train = data.iloc[train_index].reset_index(drop=True)
        valid = data.iloc[valid_index].reset_index(drop=True)
        model = TRACEModel.fit(train, spec, hyper, config, seed + fold * 17011)
        components = model.components(valid)
        for field in fields:
            assembled[field][valid_index] = components[field]
        rows.append({"fold": fold, "train_sites": int(train.siteID.nunique()), "validation_sites": int(valid.siteID.nunique()), "cross_fitted": True})
    if any(np.any(~np.isfinite(values)) for values in assembled.values()):
        raise RuntimeError("Cross-fitted stochastic layer contains non-finite component predictions")
    stochastic = estimate_stochastic_layer(data, assembled, config)
    return stochastic, pd.DataFrame(rows)

def simulate_endpoint_draws(data: pd.DataFrame, components: dict[str, np.ndarray], stochastic: dict[str, float], draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(data)
    dt = data.dt_years.to_numpy(float)
    root_dt = np.sqrt(dt)
    z1 = rng.normal(size=(draws, n))
    z2i = rng.normal(size=(draws, n))
    correlation = stochastic["endpoint_noise_correlation"]
    z2 = correlation * z1 + math.sqrt(max(1.0 - correlation * correlation, 1e-8)) * z2i
    expected_u = expected_count_from_components(data, components, "u")
    expected_v = expected_count_from_components(data, components, "v")
    observation_sd_u = stochastic["observation_scale_young"] / np.sqrt(expected_u + stochastic.get("count_offset", 0.5))
    observation_sd_v = stochastic["observation_scale_mature"] / np.sqrt(expected_v + stochastic.get("count_offset", 0.5))
    observation_u = rng.normal(size=(draws, n)) * observation_sd_u
    observation_v = rng.normal(size=(draws, n)) * observation_sd_v
    magnitude_u = np.maximum(components["magnitude_u"][None, :] + stochastic["sigma_young"] * root_dt[None, :] * z1 + observation_u, 0.0)
    magnitude_v = np.maximum(components["magnitude_v"][None, :] + stochastic["sigma_mature"] * root_dt[None, :] * z2 + observation_v, 0.0)
    positive_u = rng.random(size=(draws, n)) < components["positive_probability_u"][None, :]
    positive_v = rng.random(size=(draws, n)) < components["positive_probability_v"][None, :]
    path_u = np.where(positive_u, magnitude_u, 0.0)
    path_v = np.where(positive_v, magnitude_v, 0.0)
    return path_u, path_v, {"minimum_simulated_log_state": float(min(path_u.min(), path_v.min())), "maximum_simulated_log_state": float(max(path_u.max(), path_v.max())), "all_draws_finite": bool(np.isfinite(path_u).all() and np.isfinite(path_v).all())}

def simulate_log_diffusion_scaffold_draws(model: TRACEModel, data: pd.DataFrame, stochastic: dict[str, float], draws: int, seed: int, steps: int, config: Configuration) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model.scaffold is None:
        base_u = np.broadcast_to(data.u0_log.to_numpy(float), (draws, len(data))).copy()
        base_v = np.broadcast_to(data.v0_log.to_numpy(float), (draws, len(data))).copy()
        return base_u, base_v, {"log_diffusion_clipped_fraction": 0.0, "log_diffusion_all_finite": True}
    rng = np.random.default_rng(seed)
    n = len(data)
    x = np.broadcast_to(data.u0_log.to_numpy(float), (draws, n)).copy()
    y = np.broadcast_to(data.v0_log.to_numpy(float), (draws, n)).copy()
    dt = data.dt_years.to_numpy(float)
    step_count = max(1, int(steps))
    h = dt / step_count
    climate = model.scaffold.climate_scores(data)
    q, maturation, mortality, rho, mu, a, b = DemographicScaffold.rates(model.scaffold.parameters, climate)
    correlation = stochastic["endpoint_noise_correlation"]
    clipped = 0
    total = draws * n * step_count * 2
    for _ in range(step_count):
        u = np.expm1(np.clip(x, 0.0, config.state_log_cap))
        v = np.expm1(np.clip(y, 0.0, config.state_log_cap))
        du = q[None, :] + rho * v - (mu + maturation[None, :]) * u - a * u * u
        dv = maturation[None, :] * u - mortality[None, :] * v - b * v * v
        drift_x = du / np.maximum(1.0 + u, 1e-12)
        drift_y = dv / np.maximum(1.0 + v, 1e-12)
        z1 = rng.normal(size=(draws, n))
        z2i = rng.normal(size=(draws, n))
        z2 = correlation * z1 + math.sqrt(max(1.0 - correlation * correlation, 1e-8)) * z2i
        xn = x + h[None, :] * drift_x + stochastic["sigma_young"] * np.sqrt(h)[None, :] * z1
        yn = y + h[None, :] * drift_y + stochastic["sigma_mature"] * np.sqrt(h)[None, :] * z2
        clipped += int(np.sum((xn < 0.0) | (xn > config.state_log_cap)) + np.sum((yn < 0.0) | (yn > config.state_log_cap)))
        x = np.clip(xn, 0.0, config.state_log_cap)
        y = np.clip(yn, 0.0, config.state_log_cap)
    return x, y, {"log_diffusion_clipped_fraction": float(clipped / max(total, 1)), "log_diffusion_all_finite": bool(np.isfinite(x).all() and np.isfinite(y).all()), "log_diffusion_steps": step_count}

def simulate_log_diffusion_trace_draws(model: TRACEModel, data: pd.DataFrame, stochastic: dict[str, float], draws: int, seed: int, steps: int, config: Configuration) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed + 771)
    scaffold_u, scaffold_v, audit = simulate_log_diffusion_scaffold_draws(model, data, stochastic, draws, seed, steps, config)
    components = model.components(data)
    base_u = components["base_u"]
    base_v = components["base_v"]
    expected_u = expected_count_from_components(data, components, "u")
    expected_v = expected_count_from_components(data, components, "v")
    observation_sd_u = stochastic["observation_scale_young"] / np.sqrt(expected_u + stochastic.get("count_offset", 0.5))
    observation_sd_v = stochastic["observation_scale_mature"] / np.sqrt(expected_v + stochastic.get("count_offset", 0.5))
    magnitude_u = np.maximum(base_u[None, :] + model.hyper.gate * (scaffold_u - base_u[None, :]) + components["discrepancy_u"][None, :] + rng.normal(size=(draws, len(data))) * observation_sd_u[None, :], 0.0)
    magnitude_v = np.maximum(base_v[None, :] + model.hyper.gate * (scaffold_v - base_v[None, :]) + components["discrepancy_v"][None, :] + rng.normal(size=(draws, len(data))) * observation_sd_v[None, :], 0.0)
    positive_u = rng.random(size=(draws, len(data))) < components["positive_probability_u"][None, :]
    positive_v = rng.random(size=(draws, len(data))) < components["positive_probability_v"][None, :]
    path_u = np.where(positive_u, magnitude_u, 0.0)
    path_v = np.where(positive_v, magnitude_v, 0.0)
    audit["log_diffusion_trace_all_finite"] = bool(np.isfinite(path_u).all() and np.isfinite(path_v).all())
    return path_u, path_v, audit

def simulate_positive_state_diffusion_scaffold_draws(model: TRACEModel, data: pd.DataFrame, stochastic: dict[str, float], draws: int, seed: int, steps: int, config: Configuration) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model.scaffold is None:
        base_u = np.broadcast_to(data.u0_log.to_numpy(float), (draws, len(data))).copy()
        base_v = np.broadcast_to(data.v0_log.to_numpy(float), (draws, len(data))).copy()
        return base_u, base_v, {"positive_diffusion_truncated_fraction": 0.0, "positive_diffusion_all_finite": True}
    rng = np.random.default_rng(seed)
    n = len(data)
    u = np.broadcast_to(data.u0_kha.to_numpy(float), (draws, n)).copy()
    v = np.broadcast_to(data.v0_kha.to_numpy(float), (draws, n)).copy()
    dt = data.dt_years.to_numpy(float)
    step_count = max(1, int(steps))
    h = dt / step_count
    climate = model.scaffold.climate_scores(data)
    q, maturation, mortality, rho, mu, a, b = DemographicScaffold.rates(model.scaffold.parameters, climate)
    correlation = stochastic["endpoint_noise_correlation"]
    fraction = float(np.clip(config.positive_diffusion_demographic_fraction, 0.0, 1.0))
    sigma_u = stochastic["sigma_young"] * config.positive_diffusion_scale
    sigma_v = stochastic["sigma_mature"] * config.positive_diffusion_scale
    truncated = 0
    total = draws * n * step_count * 2
    for _ in range(step_count):
        du = q[None, :] + rho * v - (mu + maturation[None, :]) * u - a * u * u
        dv = maturation[None, :] * u - mortality[None, :] * v - b * v * v
        z1 = rng.normal(size=(draws, n))
        z2i = rng.normal(size=(draws, n))
        z2 = correlation * z1 + math.sqrt(max(1.0 - correlation * correlation, 1e-8)) * z2i
        occupancy_u = np.maximum(u / (1.0 + u), 0.0)
        occupancy_v = np.maximum(v / (1.0 + v), 0.0)
        gx = (1.0 + u) * sigma_u * np.sqrt(np.maximum(fraction * occupancy_u + (1.0 - fraction) * occupancy_u * occupancy_u, 0.0))
        gy = (1.0 + v) * sigma_v * np.sqrt(np.maximum(fraction * occupancy_v + (1.0 - fraction) * occupancy_v * occupancy_v, 0.0))
        un = u + h[None, :] * du + np.sqrt(h)[None, :] * gx * z1
        vn = v + h[None, :] * dv + np.sqrt(h)[None, :] * gy * z2
        truncated += int(np.sum(un < 0.0) + np.sum(vn < 0.0))
        u = np.maximum(un, 0.0)
        v = np.maximum(vn, 0.0)
    x = np.log1p(u)
    y = np.log1p(v)
    return x, y, {"positive_diffusion_truncated_fraction": float(truncated / max(total, 1)), "positive_diffusion_all_finite": bool(np.isfinite(x).all() and np.isfinite(y).all()), "positive_diffusion_steps": step_count, "positive_diffusion_demographic_fraction": fraction, "positive_diffusion_scale": float(config.positive_diffusion_scale)}

def simulate_positive_state_diffusion_trace_draws(model: TRACEModel, data: pd.DataFrame, stochastic: dict[str, float], draws: int, seed: int, steps: int, config: Configuration) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed + 991)
    scaffold_u, scaffold_v, audit = simulate_positive_state_diffusion_scaffold_draws(model, data, stochastic, draws, seed, steps, config)
    components = model.components(data)
    base_u = components["base_u"]
    base_v = components["base_v"]
    expected_u = expected_count_from_components(data, components, "u")
    expected_v = expected_count_from_components(data, components, "v")
    observation_sd_u = stochastic["observation_scale_young"] / np.sqrt(expected_u + stochastic.get("count_offset", 0.5))
    observation_sd_v = stochastic["observation_scale_mature"] / np.sqrt(expected_v + stochastic.get("count_offset", 0.5))
    magnitude_u = np.maximum(base_u[None, :] + model.hyper.gate * (scaffold_u - base_u[None, :]) + components["discrepancy_u"][None, :] + rng.normal(size=(draws, len(data))) * observation_sd_u[None, :], 0.0)
    magnitude_v = np.maximum(base_v[None, :] + model.hyper.gate * (scaffold_v - base_v[None, :]) + components["discrepancy_v"][None, :] + rng.normal(size=(draws, len(data))) * observation_sd_v[None, :], 0.0)
    positive_u = rng.random(size=(draws, len(data))) < components["positive_probability_u"][None, :]
    positive_v = rng.random(size=(draws, len(data))) < components["positive_probability_v"][None, :]
    path_u = np.where(positive_u, magnitude_u, 0.0)
    path_v = np.where(positive_v, magnitude_v, 0.0)
    audit["positive_diffusion_trace_all_finite"] = bool(np.isfinite(path_u).all() and np.isfinite(path_v).all())
    return path_u, path_v, audit

def diffusion_convergence_summary(coarse_u: np.ndarray, coarse_v: np.ndarray, fine_u: np.ndarray, fine_v: np.ndarray, prefix: str) -> dict[str, float]:
    rows = {}
    for key, coarse, fine in [("young", coarse_u, fine_u), ("mature", coarse_v, fine_v)]:
        coarse_mean = np.mean(coarse, axis=0)
        fine_mean = np.mean(fine, axis=0)
        coarse_sd = np.std(coarse, axis=0, ddof=1)
        fine_sd = np.std(fine, axis=0, ddof=1)
        rows[f"{prefix}_{key}_mean_abs_mean_difference"] = float(np.mean(np.abs(coarse_mean - fine_mean)))
        rows[f"{prefix}_{key}_mean_abs_sd_difference"] = float(np.mean(np.abs(coarse_sd - fine_sd)))
        rows[f"{prefix}_{key}_max_abs_mean_difference"] = float(np.max(np.abs(coarse_mean - fine_mean)))
    return rows

def raw_interval_from_draws(path_u: np.ndarray, path_v: np.ndarray, interval: float) -> dict[str, np.ndarray]:
    tail = (1.0 - interval) / 2.0
    return {"lower_u": np.quantile(path_u, tail, axis=0), "upper_u": np.quantile(path_u, 1 - tail, axis=0), "lower_v": np.quantile(path_v, tail, axis=0), "upper_v": np.quantile(path_v, 1 - tail, axis=0)}

DISTRIBUTION_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

def endpoint_distribution_summary(data: pd.DataFrame, path_u: np.ndarray, path_v: np.ndarray, seed: int, prefix: str = "distribution") -> pd.DataFrame:
    result: dict[str, np.ndarray] = {}
    for index, (key, path, base, observed) in enumerate([
        ("u", path_u, data.u0_log.to_numpy(float), data.u1_log.to_numpy(float)),
        ("v", path_v, data.v0_log.to_numpy(float), data.v1_log.to_numpy(float)),
    ]):
        original = np.expm1(path)
        result[f"{prefix}_mean_{key}_log"] = np.mean(path, axis=0)
        result[f"{prefix}_sd_{key}_log"] = np.std(path, axis=0, ddof=1)
        result[f"{prefix}_mean_{key}_kha"] = np.mean(original, axis=0)
        result[f"{prefix}_sd_{key}_kha"] = np.std(original, axis=0, ddof=1)
        for probability in DISTRIBUTION_QUANTILES:
            label = f"q{int(round(probability * 100)):02d}"
            q_log = np.quantile(path, probability, axis=0)
            result[f"{prefix}_{label}_{key}_log"] = q_log
            result[f"{prefix}_{label}_{key}_kha"] = np.expm1(q_log)
        p_zero = np.mean(path <= 1e-12, axis=0)
        p_positive = 1.0 - p_zero
        result[f"{prefix}_probability_zero_{key}"] = p_zero
        result[f"{prefix}_probability_positive_{key}"] = p_positive
        result[f"{prefix}_probability_increase_{key}"] = np.mean(path > base[None, :], axis=0)
        result[f"{prefix}_probability_decrease_{key}"] = np.mean(path < base[None, :], axis=0)
        result[f"{prefix}_crps_{key}_log"] = crps_samples(path, observed)
        result[f"{prefix}_pit_{key}"] = randomized_pit(path, observed, seed + index)
        clipped = np.clip(p_positive, 1e-12, 1.0 - 1e-12)
        result[f"{prefix}_positive_state_entropy_{key}"] = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    correlation = np.empty(path_u.shape[1], dtype=float)
    for index in range(path_u.shape[1]):
        value = np.corrcoef(path_u[:, index], path_v[:, index])[0, 1]
        correlation[index] = value if np.isfinite(value) else 0.0
    result[f"{prefix}_draw_correlation_uv"] = correlation
    return pd.DataFrame(result)

def nonconformity_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])

def aggregate_site_scores(scores: np.ndarray, sites: np.ndarray, config: Configuration) -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.DataFrame({"siteID": sites.astype(str), "score": scores})
    if config.conformal_mode == "transition":
        return scores, frame.assign(aggregate="transition")
    if config.conformal_mode == "site_max":
        grouped = frame.groupby("siteID", as_index=False).score.max()
    elif config.conformal_mode == "site_quantile":
        grouped = frame.groupby("siteID").score.quantile(config.conformal_site_quantile).reset_index()
    else:
        raise ValueError("conformal_mode must be site_max, site_quantile, or transition")
    grouped["aggregate"] = config.conformal_mode
    return grouped.score.to_numpy(float), grouped

@dataclass
class CalibrationState:
    stochastic: dict[str, float]
    conformal_q_young: float
    conformal_q_mature: float
    calibration_sites: list[str]
    stochastic_source_sites: list[str]
    score_table: pd.DataFrame
    mode: str
    stochastic_audit: pd.DataFrame


def fit_calibration_state(model: TRACEModel, calibration: pd.DataFrame, stochastic: dict[str, float], stochastic_source_sites: Sequence[str], stochastic_audit: pd.DataFrame, config: Configuration, seed: int) -> CalibrationState:
    components = model.components(calibration)
    path_u, path_v, _ = simulate_endpoint_draws(calibration, components, stochastic, config.mc_draws, seed)
    interval = raw_interval_from_draws(path_u, path_v, config.interval)
    score_u = nonconformity_scores(calibration.u1_log.to_numpy(float), interval["lower_u"], interval["upper_u"])
    score_v = nonconformity_scores(calibration.v1_log.to_numpy(float), interval["lower_v"], interval["upper_v"])
    aggregate_u, table_u = aggregate_site_scores(score_u, calibration.siteID.to_numpy(), config)
    aggregate_v, table_v = aggregate_site_scores(score_v, calibration.siteID.to_numpy(), config)
    q_u = finite_quantile(aggregate_u, config.interval)
    q_v = finite_quantile(aggregate_v, config.interval)
    table_u = table_u.rename(columns={"score": "score_young"})
    table_v = table_v.rename(columns={"score": "score_mature"})
    score_table = table_u.merge(table_v.drop(columns="aggregate", errors="ignore"), on="siteID", how="outer")
    score_table["conformal_mode"] = config.conformal_mode
    calibration_sites = sorted(calibration.siteID.astype(str).unique().tolist())
    source_sites = sorted(str(value) for value in stochastic_source_sites)
    if set(calibration_sites) & set(source_sites):
        raise RuntimeError("Conformal calibration sites overlap stochastic-estimation sites")
    return CalibrationState(stochastic, q_u, q_v, calibration_sites, source_sites, score_table, config.conformal_mode, stochastic_audit.copy())

def predictive_standardized_residuals(data: pd.DataFrame, components: dict[str, np.ndarray], stochastic: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    dt = data.dt_years.to_numpy(float)
    expected_u = expected_count_from_components(data, components, "u")
    expected_v = expected_count_from_components(data, components, "v")
    variance_u = stochastic["sigma_young"] ** 2 * dt + stochastic["observation_scale_young"] ** 2 / (expected_u + stochastic.get("count_offset", 0.5))
    variance_v = stochastic["sigma_mature"] ** 2 * dt + stochastic["observation_scale_mature"] ** 2 / (expected_v + stochastic.get("count_offset", 0.5))
    center_u = components.get("distribution_log_mean_u", components["pred_u"])
    center_v = components.get("distribution_log_mean_v", components["pred_v"])
    residual_u = (data.u1_log.to_numpy(float) - center_u) / np.sqrt(np.maximum(variance_u, 1e-10))
    residual_v = (data.v1_log.to_numpy(float) - center_v) / np.sqrt(np.maximum(variance_v, 1e-10))
    return residual_u, residual_v

def predict_with_calibration(model: TRACEModel, data: pd.DataFrame, calibration: CalibrationState, config: Configuration, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    components = model.components(data)
    path_u, path_v, audit = simulate_endpoint_draws(data, components, calibration.stochastic, config.mc_draws, seed)
    raw = raw_interval_from_draws(path_u, path_v, config.interval)
    lower_u = np.maximum(raw["lower_u"] - calibration.conformal_q_young, 0.0)
    upper_u = raw["upper_u"] + calibration.conformal_q_young
    lower_v = np.maximum(raw["lower_v"] - calibration.conformal_q_mature, 0.0)
    upper_v = raw["upper_v"] + calibration.conformal_q_mature
    point = model.point_prediction_frame(data)
    distribution = endpoint_distribution_summary(data, path_u, path_v, seed, "distribution")
    point = pd.concat([point.reset_index(drop=True), distribution.reset_index(drop=True)], axis=1)
    point["lower_u_log"] = lower_u
    point["upper_u_log"] = upper_u
    point["lower_v_log"] = lower_v
    point["upper_v_log"] = upper_v
    point["raw_lower_u_log"] = raw["lower_u"]
    point["raw_upper_u_log"] = raw["upper_u"]
    point["raw_lower_v_log"] = raw["lower_v"]
    point["raw_upper_v_log"] = raw["upper_v"]
    point["conformal_adjustment_u_log"] = calibration.conformal_q_young
    point["conformal_adjustment_v_log"] = calibration.conformal_q_mature
    point["raw_interval_width_u_log"] = raw["upper_u"] - raw["lower_u"]
    point["raw_interval_width_v_log"] = raw["upper_v"] - raw["lower_v"]
    point["calibrated_interval_width_u_log"] = upper_u - lower_u
    point["calibrated_interval_width_v_log"] = upper_v - lower_v
    residual_u, residual_v = predictive_standardized_residuals(data, components, calibration.stochastic)
    point["standardized_residual_u"] = residual_u
    point["standardized_residual_v"] = residual_v
    for key in ["u", "v"]:
        for prefix in ["lower", "upper", "raw_lower", "raw_upper"]:
            point[f"{prefix}_{key}_kha"] = np.expm1(point[f"{prefix}_{key}_log"])
    convergence_draws = min(96, max(24, config.mc_draws // 4))
    if config.log_diffusion_enabled:
        log_u, log_v, log_audit = simulate_log_diffusion_trace_draws(model, data, calibration.stochastic, config.log_diffusion_draws, seed + 1234567, config.log_diffusion_steps, config)
        log_interval = raw_interval_from_draws(log_u, log_v, config.interval)
        log_summary = endpoint_distribution_summary(data, log_u, log_v, seed + 7654321, "log_diffusion")
        point = pd.concat([point.reset_index(drop=True), log_summary.reset_index(drop=True)], axis=1)
        point["log_diffusion_lower_u_log"] = log_interval["lower_u"]
        point["log_diffusion_upper_u_log"] = log_interval["upper_u"]
        point["log_diffusion_lower_v_log"] = log_interval["lower_v"]
        point["log_diffusion_upper_v_log"] = log_interval["upper_v"]
        coarse_u, coarse_v, _ = simulate_log_diffusion_trace_draws(model, data, calibration.stochastic, convergence_draws, seed + 333333, config.log_diffusion_steps, config)
        fine_u, fine_v, _ = simulate_log_diffusion_trace_draws(model, data, calibration.stochastic, convergence_draws, seed + 333333, config.log_diffusion_steps * 2, config)
        audit.update(log_audit)
        audit.update(diffusion_convergence_summary(coarse_u, coarse_v, fine_u, fine_v, "log_diffusion_convergence"))
    if config.positive_diffusion_enabled:
        pos_u, pos_v, pos_audit = simulate_positive_state_diffusion_trace_draws(model, data, calibration.stochastic, config.positive_diffusion_draws, seed + 2234567, config.positive_diffusion_steps, config)
        pos_interval = raw_interval_from_draws(pos_u, pos_v, config.interval)
        pos_summary = endpoint_distribution_summary(data, pos_u, pos_v, seed + 8654321, "positive_diffusion")
        point = pd.concat([point.reset_index(drop=True), pos_summary.reset_index(drop=True)], axis=1)
        point["positive_diffusion_lower_u_log"] = pos_interval["lower_u"]
        point["positive_diffusion_upper_u_log"] = pos_interval["upper_u"]
        point["positive_diffusion_lower_v_log"] = pos_interval["lower_v"]
        point["positive_diffusion_upper_v_log"] = pos_interval["upper_v"]
        coarse_u, coarse_v, _ = simulate_positive_state_diffusion_trace_draws(model, data, calibration.stochastic, convergence_draws, seed + 444444, config.positive_diffusion_steps, config)
        fine_u, fine_v, _ = simulate_positive_state_diffusion_trace_draws(model, data, calibration.stochastic, convergence_draws, seed + 444444, config.positive_diffusion_steps * 2, config)
        audit.update(pos_audit)
        audit.update(diffusion_convergence_summary(coarse_u, coarse_v, fine_u, fine_v, "positive_diffusion_convergence"))
    return point, audit

def split_core_calibration(data: pd.DataFrame, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ensure_probability(fraction, "calibration_fraction")
    groups = data.siteID.astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    core, calibration = next(splitter.split(np.arange(len(data)), groups=groups))
    if set(groups[core]) & set(groups[calibration]):
        raise RuntimeError("Core/calibration site overlap detected")
    return core, calibration

def primary_spec() -> ModelSpec:
    return ModelSpec("TRACE")

def standard_ablation_specs(profile: str) -> list[ModelSpec]:
    specs = [
        ModelSpec("Discrepancy only, independently retuned", use_scaffold=False),
        ModelSpec("Without teacher smoothing", use_teacher=False),
        ModelSpec("Without realized climate", use_climate=False),
        ModelSpec("Without coordinates", use_coordinates=False),
        ModelSpec("Without variance-aware estimation", use_variance_weights=False),
        ModelSpec("Legacy sampling-precision weighting", use_precision_weights=True, use_variance_weights=False),
        ModelSpec("Without sampling-precision predictors", use_precision_features=False),
        ModelSpec("Without sampling support information", use_precision_features=False, use_variance_weights=False),
        ModelSpec("Without occurrence layer", use_occurrence=False),
    ]
    return specs[:1] if profile == "smoke" else specs

def prediction_columns_for_model(name: str) -> tuple[str, str]:
    key = slug(name)
    return f"pred_u_log__{key}", f"pred_v_log__{key}"

def derivative_frame(model: TRACEModel, data: pd.DataFrame, fold: int) -> pd.DataFrame:
    derivatives = model.discrepancy.derivatives(data, raw_scale=True)
    rows = []
    for feature_index, feature in enumerate(model.discrepancy.transform.feature_names):
        for row_index in range(len(data)):
            rows.append({
                "row_id": int(data.iloc[row_index].row_id),
                "fold": fold,
                "siteID": str(data.iloc[row_index].siteID),
                "plotID": str(data.iloc[row_index].plotID),
                "feature": feature,
                "gradient_young": float(derivatives["gradient_u"][row_index, feature_index]),
                "gradient_mature": float(derivatives["gradient_v"][row_index, feature_index]),
                "hessian_diag_young": float(derivatives["hessian_diag_u"][row_index, feature_index]),
                "hessian_diag_mature": float(derivatives["hessian_diag_v"][row_index, feature_index]),
            })
    return pd.DataFrame(rows)

def run_outer_cv(data: pd.DataFrame, config: Configuration) -> dict[str, pd.DataFrame]:
    ensure_probability(config.interval, "interval")
    groups = data.siteID.astype(str).to_numpy()
    indices = np.arange(len(data))
    n_splits = min(config.outer_folds, np.unique(groups).size)
    predictions = []
    audits = []
    parameters = []
    calibrations = []
    stochastic_audits = []
    estimation_weight_audits = []
    trace_tunings = []
    benchmark_tunings = []
    rates = []
    pcas = []
    derivatives = []
    hessian_pairs = []
    model_selections = []
    for fold, (train_index, test_index) in enumerate(GroupKFold(n_splits).split(indices, groups=groups), 1):
        outer_train = data.iloc[train_index].reset_index(drop=True)
        test = data.iloc[test_index].reset_index(drop=True)
        core_index, calibration_index = split_core_calibration(outer_train, config.calibration_fraction, config.seed + fold * 101)
        core = outer_train.iloc[core_index].reset_index(drop=True)
        calibration = outer_train.iloc[calibration_index].reset_index(drop=True)
        if not validate_no_group_overlap(core, calibration, test):
            raise RuntimeError(f"site leakage in fold {fold}")
        main_spec = primary_spec()
        main_hyper, main_tuning = tune_trace_model(core, main_spec, config, config.seed + fold * 1000003)
        if not main_tuning.empty:
            main_tuning.insert(0, "outer_fold", fold)
            trace_tunings.append(main_tuning)
        main_model = TRACEModel.fit(core, main_spec, main_hyper, config, config.seed + fold * 100003)
        stochastic, stochastic_audit = fit_stochastic_layer_cross_fitted(core, main_spec, main_hyper, config, config.seed + fold * 100003 + 30011)
        stochastic_audit.insert(0, "outer_fold", fold)
        stochastic_audits.append(stochastic_audit)
        calibration_state = fit_calibration_state(main_model, calibration, stochastic, core.siteID.astype(str).unique().tolist(), stochastic_audit, config, config.seed + fold * 100003 + 50003)
        prediction, draw_audit = predict_with_calibration(main_model, test, calibration_state, config, config.seed + fold * 100003 + 90001)
        keep = [
            "row_id", "transition_id", "siteID", "plotID", "dt_years", "log_dt", "target_year", "event_year", "decimalLatitude", "decimalLongitude",
            "u0_kha", "v0_kha", "u1_kha", "v1_kha", "u0_log", "v0_log", "u1_log", "v1_log", "zero_u0", "zero_v0", "positive_u1", "positive_v1",
            "young_baseline_area_precision", "mature_baseline_area_precision", "young_baseline_area_missing", "mature_baseline_area_missing", "young_area_precision", "mature_area_precision", "area_young_t0_m2_reconstructed", "area_mature_t0_m2_reconstructed", "area_young_m2", "area_mature_m2", "young_count_t0", "mature_count_t0", "young_count_t1", "mature_count_t1", "t0_date", "t1_date", "t0_year", "t1_year", "midpoint_year", "sequence_index_plot", "n_transitions_plot", *RAW_CLIMATE_MODEL,
        ]
        out = test[keep].copy()
        out.insert(1, "fold", fold)
        primary_u, primary_v = prediction_columns_for_model(main_spec.name)
        out[primary_u] = prediction.pred_u_log.to_numpy(float)
        out[primary_v] = prediction.pred_v_log.to_numpy(float)
        for column in prediction.columns:
            if column not in {"pred_u_log", "pred_v_log"}:
                out[column] = prediction[column].to_numpy()
        persistence_u, persistence_v = prediction_columns_for_model("Persistence")
        out[persistence_u] = out.u0_log
        out[persistence_v] = out.v0_log
        scaffold_u, scaffold_v = prediction_columns_for_model("Demographic scaffold only")
        if main_model.scaffold is not None:
            scaffold_prediction_u, scaffold_prediction_v = main_model.scaffold.predict(test)
            out[scaffold_u] = scaffold_prediction_u
            out[scaffold_v] = scaffold_prediction_v
        else:
            out[scaffold_u] = out.u0_log
            out[scaffold_v] = out.v0_log
        model_selections.append({"outer_fold": fold, "model": main_spec.name, **hyperparameter_record(main_hyper), "type": "TRACE"})
        weight_table = main_model.estimation_weight_audit.copy()
        if not weight_table.empty:
            weight_table.insert(0, "outer_fold", fold)
            weight_table.insert(1, "model", main_spec.name)
            estimation_weight_audits.append(weight_table)
        for index, ablation_spec in enumerate(standard_ablation_specs(config.profile), 1):
            hyper, tuning = tune_trace_model(core, ablation_spec, config, config.seed + fold * 1000003 + index * 70001)
            if not tuning.empty:
                tuning.insert(0, "outer_fold", fold)
                trace_tunings.append(tuning)
            ablation_model = TRACEModel.fit(core, ablation_spec, hyper, config, config.seed + fold * 100003 + index * 7001)
            components = ablation_model.components(test)
            column_u, column_v = prediction_columns_for_model(ablation_spec.name)
            out[column_u] = components["pred_u"]
            out[column_v] = components["pred_v"]
            model_selections.append({"outer_fold": fold, "model": ablation_spec.name, **hyperparameter_record(hyper), "type": "TRACE_ablation"})
        benchmark_spec = ModelSpec("benchmark")
        for index, name in enumerate(["Extra Trees direct", "Histogram boosting direct", "Ridge direct", "Hurdle Extra Trees"], 1):
            parameters_selected, tuning = tune_direct_benchmark(core, name, benchmark_spec, config, config.seed + fold * 700001 + index * 13001)
            if not tuning.empty:
                tuning.insert(0, "outer_fold", fold)
                benchmark_tunings.append(tuning)
            benchmark = DirectBenchmark.fit(name, core, benchmark_spec, config, config.seed + fold * 70001 + index * 1301, parameters_selected)
            pred_u, pred_v = benchmark.predict(test)
            column_u, column_v = prediction_columns_for_model(name)
            out[column_u] = pred_u
            out[column_v] = pred_v
            model_selections.append({"outer_fold": fold, "model": name, "gate": np.nan, "ridge_scale": np.nan, "type": "benchmark", "parameters": json.dumps(parameters_selected, sort_keys=True)})
        predictions.append(out)
        derivatives.append(derivative_frame(main_model, test, fold))
        pairs = main_model.discrepancy.mean_abs_hessian_pairs(test, raw_scale=True)
        pairs.insert(0, "outer_fold", fold)
        hessian_pairs.append(pairs)
        site_sets = [set(frame.siteID.astype(str)) for frame in (outer_train, core, calibration, test)]
        audits.append({
            "fold": fold,
            "outer_train_rows": len(outer_train),
            "core_rows": len(core),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "outer_train_sites": len(site_sets[0]),
            "core_sites": len(site_sets[1]),
            "calibration_sites": len(site_sets[2]),
            "test_sites": len(site_sets[3]),
            "train_test_site_overlap": len(site_sets[0] & site_sets[3]),
            "core_calibration_site_overlap": len(site_sets[1] & site_sets[2]),
            "core_test_site_overlap": len(site_sets[1] & site_sets[3]),
            "calibration_test_site_overlap": len(site_sets[2] & site_sets[3]),
            "stochastic_calibration_site_overlap": len(set(calibration_state.stochastic_source_sites) & set(calibration_state.calibration_sites)),
            "selected_gate": main_hyper.gate,
            "selected_ridge_scale": main_hyper.ridge_scale,
            "selected_distillation_weight": main_hyper.distillation_weight,
            "selected_occurrence_gate": main_hyper.occurrence_gate,
            "selected_gamma_scale": main_hyper.gamma_scale,
            "selected_kernel_alpha": main_hyper.kernel_alpha,
            "conformal_mode": config.conformal_mode,
            "conformal_q_young": calibration_state.conformal_q_young,
            "conformal_q_mature": calibration_state.conformal_q_mature,
            **draw_audit,
            **main_model.discrepancy.mathematical_audit(),
            **{f"stochastic_{key}": value for key, value in calibration_state.stochastic.items()},
            **{f"occurrence_{key}": value for key, value in (main_model.occurrence.audit() if main_model.occurrence is not None else {}).items()},
        })
        if main_model.scaffold is not None:
            parameters.append(main_model.scaffold.parameter_table(f"outer_fold_{fold}"))
            rates.append(climate_rate_response(main_model.scaffold, f"outer_fold_{fold}"))
            if main_model.scaffold.climate is not None:
                pcas.append(main_model.scaffold.climate.loading_table(f"outer_fold_{fold}_scaffold"))
        if main_model.discrepancy.transform.climate is not None:
            pcas.append(main_model.discrepancy.transform.climate.loading_table(f"outer_fold_{fold}_discrepancy"))
        calibration_table = calibration_state.score_table.copy()
        calibration_table.insert(0, "fold", fold)
        calibration_table["q_young"] = calibration_state.conformal_q_young
        calibration_table["q_mature"] = calibration_state.conformal_q_mature
        calibrations.append(calibration_table)
    oof = pd.concat(predictions, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    if len(oof) != len(data) or oof.row_id.duplicated().any():
        raise RuntimeError("outer CV failed")
    return {
        "OOF_Predictions": oof,
        "CV_Audit": pd.DataFrame(audits),
        "Fold_Parameters": pd.concat(parameters, ignore_index=True) if parameters else pd.DataFrame(),
        "Calibration_Scores": pd.concat(calibrations, ignore_index=True) if calibrations else pd.DataFrame(),
        "Stochastic_Fit_Audit": pd.concat(stochastic_audits, ignore_index=True) if stochastic_audits else pd.DataFrame(),
        "Estimation_Weight_Audit": pd.concat(estimation_weight_audits, ignore_index=True) if estimation_weight_audits else pd.DataFrame(),
        "TRACE_Tuning": pd.concat(trace_tunings, ignore_index=True) if trace_tunings else pd.DataFrame(),
        "Benchmark_Tuning": pd.concat(benchmark_tunings, ignore_index=True) if benchmark_tunings else pd.DataFrame(),
        "Model_Selections": pd.DataFrame(model_selections),
        "Fold_Climate_Rates": pd.concat(rates, ignore_index=True) if rates else pd.DataFrame(),
        "Fold_PCA_Loadings": pd.concat(pcas, ignore_index=True) if pcas else pd.DataFrame(),
        "OOF_Derivatives": pd.concat(derivatives, ignore_index=True) if derivatives else pd.DataFrame(),
        "OOF_Hessian_Interactions": pd.concat(hessian_pairs, ignore_index=True) if hessian_pairs else pd.DataFrame(),
    }

def discover_models(oof: pd.DataFrame) -> dict[str, tuple[str, str]]:
    models = {}
    for column in oof.columns:
        if column.startswith("pred_u_log__"):
            key = column.replace("pred_u_log__", "")
            v_column = f"pred_v_log__{key}"
            if v_column in oof.columns:
                models[key] = (column, v_column)
    return models

def model_label_from_key(key: str) -> str:
    mapping = {
        "trace": "TRACE",
        "persistence": "Persistence",
        "demographic_scaffold_only": "Demographic scaffold only",
        "discrepancy_only_independently_retuned": "Discrepancy only, independently retuned",
        "without_teacher_smoothing": "Without teacher smoothing",
        "without_realized_climate": "Without realized climate",
        "without_coordinates": "Without coordinates",
        "without_variance_aware_estimation": "Without variance-aware estimation",
        "legacy_sampling_precision_weighting": "Legacy sampling-precision weighting",
        "without_sampling_precision_predictors": "Without sampling-precision predictors",
        "without_sampling_support_information": "Without sampling support information",
        "without_occurrence_layer": "Without occurrence layer",
        "extra_trees_direct": "Extra Trees direct",
        "histogram_boosting_direct": "Histogram boosting direct",
        "ridge_direct": "Ridge direct",
        "hurdle_extra_trees": "Hurdle Extra Trees",
    }
    return mapping.get(key, key.replace("_", " ").title())

def metric_values(y: np.ndarray, prediction: np.ndarray, lower: np.ndarray | None = None, upper: np.ndarray | None = None, interval: float = 0.90) -> dict[str, float]:
    values = {
        "RMSE_log1p": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE_log1p": float(mean_absolute_error(y, prediction)),
        "R2_log1p": float(r2_score(y, prediction)),
        "Spearman": safe_spearman(y, prediction),
        "Bias_log1p": float(np.mean(prediction - y)),
    }
    if lower is not None and upper is not None:
        values.update({
            "Coverage": float(np.mean((y >= lower) & (y <= upper))),
            "Mean_interval_width_log1p": float(np.mean(upper - lower)),
            "Median_interval_width_log1p": float(np.median(upper - lower)),
            "Interval_score": interval_score(y, lower, upper, interval),
        })
    return values

def site_bootstrap_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    sites = frame.siteID.astype(str).unique()
    sampled = rng.choice(sites, size=len(sites), replace=True)
    pieces = []
    for replicate, site in enumerate(sampled):
        idx = np.flatnonzero(frame.siteID.astype(str).to_numpy() == site)
        pieces.append(idx)
    return np.concatenate(pieces) if pieces else np.array([], dtype=int)

def bootstrap_statistic(frame: pd.DataFrame, statistic: Any, repetitions: int, seed: int) -> np.ndarray:
    if repetitions <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        index = site_bootstrap_indices(frame, rng)
        value = statistic(frame.iloc[index])
        if np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)

def bootstrap_interval(distribution: np.ndarray) -> tuple[float, float]:
    finite = distribution[np.isfinite(distribution)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))

def signflip_test(site_differences: np.ndarray, permutations: int, seed: int) -> float:
    d = np.asarray(site_differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    observed = abs(float(np.mean(d)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(max(1, permutations)):
        statistic = abs(float(np.mean(d * rng.choice([-1.0, 1.0], size=len(d)))))
        exceed += statistic >= observed - 1e-15
    return float((exceed + 1) / (max(1, permutations) + 1))

def model_metrics_table(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    rows = []
    models = discover_models(oof)
    for key, columns in models.items():
        label = model_label_from_key(key)
        for stage, target, prediction_column in [("young", "u1_log", columns[0]), ("mature", "v1_log", columns[1])]:
            y = oof[target].to_numpy(float)
            prediction = oof[prediction_column].to_numpy(float)
            metrics = metric_values(y, prediction)
            for metric, estimate in metrics.items():
                def statistic(frame: pd.DataFrame, target=target, prediction_column=prediction_column, metric=metric) -> float:
                    values = metric_values(frame[target].to_numpy(float), frame[prediction_column].to_numpy(float))
                    return values[metric]
                distribution = bootstrap_statistic(oof, statistic, config.cluster_bootstrap, config.seed + stable_int_seed(label, stage, metric) % 1000000)
                low, high = bootstrap_interval(distribution)
                rows.append({"model": label, "stage": stage, "metric": metric, "estimate": estimate, "CI_low": low, "CI_high": high, "bootstrap_repetitions": len(distribution)})
    return pd.DataFrame(rows)

def paired_model_comparisons(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    models = discover_models(oof)
    primary_key = slug(primary_spec().name)
    if primary_key not in models:
        return pd.DataFrame()
    rows = []
    primary_columns = models[primary_key]
    for comparison_key, comparison_columns in models.items():
        if comparison_key == primary_key:
            continue
        comparison_label = model_label_from_key(comparison_key)
        for stage, target, primary_column, comparison_column in [
            ("young", "u1_log", primary_columns[0], comparison_columns[0]),
            ("mature", "v1_log", primary_columns[1], comparison_columns[1]),
        ]:
            y = oof[target].to_numpy(float)
            rmse_primary = float(np.sqrt(mean_squared_error(y, oof[primary_column])))
            rmse_comparison = float(np.sqrt(mean_squared_error(y, oof[comparison_column])))
            observed_difference = rmse_comparison - rmse_primary
            def statistic(frame: pd.DataFrame, target=target, primary_column=primary_column, comparison_column=comparison_column) -> float:
                yy = frame[target].to_numpy(float)
                rp = np.sqrt(mean_squared_error(yy, frame[primary_column]))
                rc = np.sqrt(mean_squared_error(yy, frame[comparison_column]))
                return float(rc - rp)
            distribution = bootstrap_statistic(oof, statistic, config.paired_bootstrap, config.seed + stable_int_seed(comparison_label, stage) % 1000000)
            low, high = bootstrap_interval(distribution)
            probability_better = float(np.mean(distribution > 0)) if distribution.size else float("nan")
            site_differences = []
            for _, group in oof.groupby("siteID"):
                yy = group[target].to_numpy(float)
                mse_primary = float(np.mean((group[primary_column].to_numpy(float) - yy) ** 2))
                mse_comparison = float(np.mean((group[comparison_column].to_numpy(float) - yy) ** 2))
                site_differences.append(mse_comparison - mse_primary)
            p_sign = signflip_test(np.asarray(site_differences), config.signflip_permutations, config.seed + stable_int_seed("sign", comparison_label, stage) % 1000000)
            rows.append({
                "stage": stage,
                "primary_model": primary_spec().name,
                "comparison_model": comparison_label,
                "primary_RMSE": rmse_primary,
                "comparison_RMSE": rmse_comparison,
                "RMSE_difference_comparison_minus_primary": observed_difference,
                "CI_low": low,
                "CI_high": high,
                "probability_primary_better": probability_better,
                "site_equal_MSE_signflip_p": p_sign,
            })
    return pd.DataFrame(rows)

def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    models = discover_models(oof)
    rows = []
    for fold, frame in oof.groupby("fold"):
        for key, columns in models.items():
            label = model_label_from_key(key)
            for stage, target, prediction in [("young", "u1_log", columns[0]), ("mature", "v1_log", columns[1])]:
                values = metric_values(frame[target].to_numpy(float), frame[prediction].to_numpy(float))
                for metric, estimate in values.items():
                    rows.append({"fold": int(fold), "model": label, "stage": stage, "metric": metric, "estimate": estimate})
    return pd.DataFrame(rows)

def grouped_skill(oof: pd.DataFrame, group: str) -> pd.DataFrame:
    primary_u, primary_v = prediction_columns_for_model(primary_spec().name)
    persistence_u, persistence_v = prediction_columns_for_model("Persistence")
    rows = []
    for value, frame in oof.groupby(group):
        for stage, target, primary, persistence in [
            ("young", "u1_log", primary_u, persistence_u),
            ("mature", "v1_log", primary_v, persistence_v),
        ]:
            y = frame[target].to_numpy(float)
            rmse_primary = float(np.sqrt(mean_squared_error(y, frame[primary])))
            rmse_persistence = float(np.sqrt(mean_squared_error(y, frame[persistence])))
            rows.append({group: value, "stage": stage, "transitions": len(frame), "RMSE_TRACE": rmse_primary, "RMSE_persistence": rmse_persistence, "skill_vs_persistence": 1.0 - rmse_primary / rmse_persistence if rmse_persistence > 1e-12 else np.nan})
    return pd.DataFrame(rows)

def site_equal_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    models = discover_models(oof)
    rows = []
    for key, columns in models.items():
        label = model_label_from_key(key)
        for stage, target, prediction in [("young", "u1_log", columns[0]), ("mature", "v1_log", columns[1])]:
            site_rmse = []
            site_mse = []
            for _, frame in oof.groupby("siteID"):
                error = frame[prediction].to_numpy(float) - frame[target].to_numpy(float)
                site_rmse.append(float(np.sqrt(np.mean(error * error))))
                site_mse.append(float(np.mean(error * error)))
            rows.append({"model": label, "stage": stage, "mean_site_RMSE": float(np.mean(site_rmse)), "median_site_RMSE": float(np.median(site_rmse)), "site_equal_RMSE_from_MSE": float(np.sqrt(np.mean(site_mse))), "sites": len(site_rmse)})
    result = pd.DataFrame(rows)
    for stage in ["young", "mature"]:
        persistence = result[(result.model == "Persistence") & (result.stage == stage)]
        if persistence.empty:
            continue
        reference = float(persistence.site_equal_RMSE_from_MSE.iloc[0])
        mask = result.stage == stage
        result.loc[mask, "site_equal_skill_vs_persistence"] = 1.0 - result.loc[mask, "site_equal_RMSE_from_MSE"] / reference
    return result

def horizon_skill(oof: pd.DataFrame) -> pd.DataFrame:
    quantiles = np.unique(np.quantile(oof.dt_years.to_numpy(float), [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    if len(quantiles) < 3:
        return pd.DataFrame()
    frame = oof.copy()
    frame["horizon_bin"] = pd.cut(frame.dt_years, bins=quantiles, include_lowest=True, duplicates="drop")
    return grouped_skill(frame.dropna(subset=["horizon_bin"]), "horizon_bin")

def ood_analysis(oof: pd.DataFrame) -> pd.DataFrame:
    if "nearest_feature_distance" not in oof:
        return pd.DataFrame()
    frame = oof.copy()
    try:
        frame["distance_bin"] = pd.qcut(frame.nearest_feature_distance, q=5, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    result = grouped_skill(frame, "distance_bin")
    distance = frame.groupby("distance_bin", observed=True).nearest_feature_distance.agg(["median", "min", "max"]).reset_index()
    return result.merge(distance, on="distance_bin", how="left")

def transition_regime_analysis(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    primary = prediction_columns_for_model(primary_spec().name)
    persistence = prediction_columns_for_model("Persistence")
    rows = []
    for index, (stage, key) in enumerate([("young", "u"), ("mature", "v")]):
        start = oof[f"zero_{key}0"].to_numpy(int) == 0
        end = oof[f"positive_{key}1"].to_numpy(int) == 1
        labels = np.select([~start & ~end, ~start & end, start & ~end, start & end], ["zero_to_zero", "zero_to_positive", "positive_to_zero", "positive_to_positive"], default="unknown")
        for regime in np.unique(labels):
            mask = labels == regime
            subset = oof.loc[mask].copy()
            y = subset[f"{key}1_log"].to_numpy(float)
            prediction = subset[primary[index]].to_numpy(float)
            reference = subset[persistence[index]].to_numpy(float)
            rmse_model = float(np.sqrt(mean_squared_error(y, prediction)))
            rmse_reference = float(np.sqrt(mean_squared_error(y, reference)))
            skill = 1.0 - rmse_model / rmse_reference if rmse_reference > 1e-12 else np.nan
            def statistic(frame: pd.DataFrame, key=key, pcol=primary[index], rcol=persistence[index]) -> float:
                yy = frame[f"{key}1_log"].to_numpy(float)
                rp = float(np.sqrt(mean_squared_error(yy, frame[pcol])))
                rr = float(np.sqrt(mean_squared_error(yy, frame[rcol])))
                return 1.0 - rp / rr if rr > 1e-12 else np.nan
            distribution = bootstrap_statistic(subset, statistic, config.regime_bootstrap, config.seed + 40000 + index * 1000 + sum(ord(c) for c in regime))
            low, high = bootstrap_interval(distribution)
            probability_positive = float(np.mean(distribution > 0)) if distribution.size else np.nan
            rows.append({
                "stage": stage,
                "regime": regime,
                "transitions": int(mask.sum()),
                "sites": int(subset.siteID.nunique()),
                "TRACE_RMSE": rmse_model,
                "Persistence_RMSE": rmse_reference,
                "skill_vs_persistence": skill,
                "skill_CI_low": low,
                "skill_CI_high": high,
                "bootstrap_probability_positive_skill": probability_positive,
            })
    return pd.DataFrame(rows)

def occurrence_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        y = oof[f"positive_{key}1"].to_numpy(int)
        probability = np.clip(oof[f"positive_probability_{key}"].to_numpy(float), 1e-8, 1 - 1e-8)
        rows.append({
            "stage": stage,
            "ROC_AUC": float(roc_auc_score(y, probability)) if np.unique(y).size > 1 else np.nan,
            "Brier": float(brier_score_loss(y, probability)),
            "Log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "prevalence": float(np.mean(y)),
        })
    return pd.DataFrame(rows)

def occurrence_reliability(oof: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        y = oof[f"positive_{key}1"].to_numpy(int)
        p = oof[f"positive_probability_{key}"].to_numpy(float)
        edges = np.linspace(0, 1, bins + 1)
        labels = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
        for b in range(bins):
            mask = labels == b
            if mask.sum() == 0:
                continue
            rows.append({"stage": stage, "bin": b, "count": int(mask.sum()), "mean_predicted_probability": float(np.mean(p[mask])), "observed_positive_frequency": float(np.mean(y[mask]))})
    return pd.DataFrame(rows)

def occurrence_calibration_summary(oof: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        y = oof[f"positive_{key}1"].to_numpy(int)
        p = np.clip(oof[f"positive_probability_{key}"].to_numpy(float), 1e-6, 1 - 1e-6)
        if np.unique(y).size > 1:
            predictor = logit(p).reshape(-1, 1)
            model = LogisticRegression(C=1e6, max_iter=5000).fit(predictor, y)
            intercept = float(model.intercept_[0])
            slope = float(model.coef_[0, 0])
        else:
            intercept = np.nan
            slope = np.nan
        edges = np.linspace(0.0, 1.0, bins + 1)
        labels = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
        ece = 0.0
        maximum_gap = 0.0
        for b in range(bins):
            mask = labels == b
            if not mask.any():
                continue
            gap = abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
            ece += float(mask.mean()) * gap
            maximum_gap = max(maximum_gap, gap)
        rows.append({"stage": stage, "calibration_intercept": intercept, "calibration_slope": slope, "ECE": float(ece), "maximum_bin_gap": float(maximum_gap), "bins": bins, "prevalence": float(np.mean(y)), "mean_probability": float(np.mean(p))})
    return pd.DataFrame(rows)

def zero_origin_support_sensitivity(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    primary = prediction_columns_for_model(primary_spec().name)
    persistence = prediction_columns_for_model("Persistence")
    rows = []
    tolerance = max(float(config.support_change_tolerance), 0.0)
    for index, (stage, key) in enumerate([("young", "u"), ("mature", "v")]):
        t0 = pd.to_numeric(oof[f"area_{stage}_t0_m2_reconstructed"], errors="coerce").to_numpy(float)
        t1 = pd.to_numeric(oof[f"area_{stage}_m2"], errors="coerce").to_numpy(float)
        known = np.isfinite(t0) & (t0 > 0)
        relative_change = np.full(len(oof), np.nan, dtype=float)
        relative_change[known] = np.abs(t1[known] - t0[known]) / np.maximum(t0[known], 1e-12)
        comparable = known & np.isfinite(relative_change) & (relative_change <= tolerance)
        changed = known & np.isfinite(relative_change) & (relative_change > tolerance)
        zero_positive = (oof[f"zero_{key}0"].to_numpy(int) == 1) & (oof[f"positive_{key}1"].to_numpy(int) == 1)
        subsets = {
            "all_zero_to_positive": zero_positive,
            "baseline_support_known": zero_positive & known,
            "baseline_support_unknown": zero_positive & ~known,
            "support_comparable": zero_positive & comparable,
            "support_changed": zero_positive & changed,
        }
        for label, mask in subsets.items():
            subset = oof.loc[mask].copy()
            if subset.empty:
                continue
            y = subset[f"{key}1_log"].to_numpy(float)
            pred = subset[primary[index]].to_numpy(float)
            ref = subset[persistence[index]].to_numpy(float)
            rmse_model = float(np.sqrt(mean_squared_error(y, pred)))
            rmse_reference = float(np.sqrt(mean_squared_error(y, ref)))
            skill = 1.0 - rmse_model / rmse_reference if rmse_reference > 1e-12 else np.nan
            if subset.siteID.nunique() >= 2 and rmse_reference > 1e-12:
                def statistic(frame: pd.DataFrame, key=key, pcol=primary[index], rcol=persistence[index]) -> float:
                    yy = frame[f"{key}1_log"].to_numpy(float)
                    rp = float(np.sqrt(mean_squared_error(yy, frame[pcol].to_numpy(float))))
                    rr = float(np.sqrt(mean_squared_error(yy, frame[rcol].to_numpy(float))))
                    return 1.0 - rp / rr if rr > 1e-12 else np.nan
                distribution = bootstrap_statistic(subset, statistic, config.regime_bootstrap, config.seed + 880000 + index * 10000 + stable_int_seed(label) % 10000)
                low, high = bootstrap_interval(distribution)
                probability = float(np.mean(distribution > 0)) if distribution.size else np.nan
            else:
                low, high, probability = np.nan, np.nan, np.nan
            rows.append({"stage": stage, "subset": label, "transitions": len(subset), "sites": int(subset.siteID.nunique()), "support_change_tolerance": tolerance, "TRACE_RMSE": rmse_model, "Persistence_RMSE": rmse_reference, "skill_vs_persistence": skill, "skill_CI_low": low, "skill_CI_high": high, "bootstrap_probability_positive_skill": probability})
    return pd.DataFrame(rows)

def interval_calibration_diagnostics(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        y = oof[f"{key}1_log"].to_numpy(float)
        for label, lower_col, upper_col in [
            ("raw", f"raw_lower_{key}_log", f"raw_upper_{key}_log"),
            ("site_conformal", f"lower_{key}_log", f"upper_{key}_log"),
            ("log_diffusion_sensitivity", f"log_diffusion_lower_{key}_log", f"log_diffusion_upper_{key}_log"),
            ("positive_diffusion_sensitivity", f"positive_diffusion_lower_{key}_log", f"positive_diffusion_upper_{key}_log"),
        ]:
            if lower_col not in oof or upper_col not in oof:
                continue
            lower = oof[lower_col].to_numpy(float)
            upper = oof[upper_col].to_numpy(float)
            covered = (y >= lower) & (y <= upper)
            site_coverage = pd.DataFrame({"siteID": oof.siteID.astype(str), "covered": covered}).groupby("siteID").covered.mean()
            simultaneous = pd.DataFrame({"siteID": oof.siteID.astype(str), "covered": covered}).groupby("siteID").covered.all()
            rows.append({
                "stage": stage,
                "interval": label,
                "nominal": config.interval,
                "transition_coverage": float(np.mean(covered)),
                "mean_site_coverage": float(site_coverage.mean()),
                "median_site_coverage": float(site_coverage.median()),
                "simultaneous_site_coverage": float(simultaneous.mean()),
                "mean_width": float(np.mean(upper - lower)),
                "median_width": float(np.median(upper - lower)),
                "interval_score": interval_score(y, lower, upper, config.interval),
            })
    return pd.DataFrame(rows)

def cluster_pit_statistic(frame: pd.DataFrame, pit_column: str) -> tuple[float, float, float]:
    grid = np.linspace(0.05, 0.95, 19)
    site_curves = []
    site_means = []
    for _, group in frame.groupby("siteID"):
        pit = np.clip(group[pit_column].to_numpy(float), 0.0, 1.0)
        pit = pit[np.isfinite(pit)]
        if pit.size == 0:
            continue
        site_curves.append(np.array([np.mean(pit <= value) for value in grid], dtype=float))
        site_means.append(float(np.mean(pit)))
    if not site_curves:
        return np.nan, np.nan, np.nan
    curves = np.vstack(site_curves)
    mean_curve = np.mean(curves, axis=0)
    cvm = float(np.mean((mean_curve - grid) ** 2))
    max_deviation = float(np.max(np.abs(mean_curve - grid)))
    mean_site_pit = float(np.mean(site_means))
    return cvm, max_deviation, mean_site_pit

def cluster_pit_bootstrap(frame: pd.DataFrame, pit_column: str, repetitions: int, seed: int) -> tuple[float, float, float, float]:
    observed, _, _ = cluster_pit_statistic(frame, pit_column)
    sites = frame.siteID.astype(str).unique()
    if len(sites) < 2 or repetitions <= 0:
        return observed, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(sites, size=len(sites), replace=True)
        pieces = []
        for index, site in enumerate(sampled):
            piece = frame[frame.siteID.astype(str) == site].copy()
            piece["siteID"] = f"{site}__{index}"
            pieces.append(piece)
        statistic, _, _ = cluster_pit_statistic(pd.concat(pieces, ignore_index=True), pit_column)
        if np.isfinite(statistic):
            values.append(statistic)
    if not values:
        return observed, np.nan, np.nan, np.nan
    array = np.asarray(values, dtype=float)
    return observed, float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975)), float(np.std(array, ddof=1))

def pit_cluster_bins(oof: pd.DataFrame, config: Configuration, bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    repetitions = min(config.cluster_bootstrap, 500)
    for stage, key in [("young", "u"), ("mature", "v")]:
        for prefix in ["distribution", "log_diffusion", "positive_diffusion"]:
            pit_column = f"{prefix}_pit_{key}"
            if pit_column not in oof:
                continue
            site_vectors = []
            sites = []
            for site, group in oof.groupby("siteID"):
                pit = np.clip(group[pit_column].to_numpy(float), 0.0, 1.0)
                counts, _ = np.histogram(pit, bins=edges)
                site_vectors.append(counts / max(counts.sum(), 1))
                sites.append(str(site))
            matrix = np.vstack(site_vectors)
            rng = np.random.default_rng(stable_int_seed(config.seed, "pit_bins", stage, prefix))
            bootstrap = np.empty((max(repetitions, 1), bins), dtype=float)
            for b in range(max(repetitions, 1)):
                index = rng.integers(0, len(matrix), size=len(matrix))
                bootstrap[b] = np.mean(matrix[index], axis=0)
            mean = np.mean(matrix, axis=0)
            for b in range(bins):
                rows.append({"stage": stage, "distribution": prefix, "bin": b, "lower": edges[b], "upper": edges[b + 1], "site_equal_frequency": float(mean[b]), "bootstrap_CI_low": float(np.quantile(bootstrap[:, b], 0.025)), "bootstrap_CI_high": float(np.quantile(bootstrap[:, b], 0.975)), "uniform_reference": float(1.0 / bins), "sites": len(sites)})
    return pd.DataFrame(rows)

def distribution_diagnostics(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    rows = []
    repetitions = min(config.cluster_bootstrap, 1000)
    for stage, key in [("young", "u"), ("mature", "v")]:
        for prefix in ["distribution", "log_diffusion", "positive_diffusion"]:
            crps_col = f"{prefix}_crps_{key}_log"
            pit_col = f"{prefix}_pit_{key}"
            if crps_col not in oof or pit_col not in oof:
                continue
            pit = np.clip(oof[pit_col].to_numpy(float), 0.0, 1.0)
            cvm, max_deviation, mean_site_pit = cluster_pit_statistic(oof[["siteID", pit_col]].copy(), pit_col)
            _, ci_low, ci_high, bootstrap_sd = cluster_pit_bootstrap(oof[["siteID", pit_col]].copy(), pit_col, repetitions, stable_int_seed(config.seed, "pit", stage, prefix))
            rows.append({"stage": stage, "distribution": prefix, "mean_CRPS_log": float(oof[crps_col].mean()), "median_CRPS_log": float(oof[crps_col].median()), "transition_weighted_PIT_mean": float(np.mean(pit)), "transition_weighted_PIT_variance": float(np.var(pit, ddof=1)), "site_equal_PIT_mean": mean_site_pit, "site_equal_CvM": cvm, "site_equal_max_CDF_deviation": max_deviation, "cluster_bootstrap_CvM_CI_low": ci_low, "cluster_bootstrap_CvM_CI_high": ci_high, "cluster_bootstrap_CvM_sd": bootstrap_sd, "formal_iid_uniformity_p_value_reported": False})
    return pd.DataFrame(rows)

def residual_serial_statistic(frame: pd.DataFrame, residual_column: str) -> tuple[float, int]:
    left = []
    right = []
    for _, group in frame.sort_values(["siteID", "plotID", "event_year"]).groupby(["siteID", "plotID"]):
        values = group[residual_column].to_numpy(float)
        if len(values) >= 2:
            left.extend(values[:-1])
            right.extend(values[1:])
    if len(left) < 3:
        return float("nan"), len(left)
    correlation = np.corrcoef(np.asarray(left), np.asarray(right))[0, 1]
    return float(correlation) if np.isfinite(correlation) else float("nan"), len(left)

def residual_serial_permutation_p(frame: pd.DataFrame, residual_column: str, observed: float, repetitions: int, seed: int) -> float:
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    exceed = 0
    base = frame[["siteID", "plotID", "event_year", residual_column]].copy()
    for _ in range(max(1, repetitions)):
        permuted = base.copy()
        values = permuted[residual_column].to_numpy(float).copy()
        for _, indices in permuted.groupby("siteID").groups.items():
            idx = np.asarray(list(indices), dtype=int)
            values[idx] = rng.permutation(values[idx])
        permuted[residual_column] = values
        statistic, _ = residual_serial_statistic(permuted, residual_column)
        exceed += np.isfinite(statistic) and abs(statistic) >= abs(observed) - 1e-15
    return float((exceed + 1) / (max(1, repetitions) + 1))

def residual_diagnostics(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    rows = []
    primary_u, primary_v = prediction_columns_for_model(primary_spec().name)
    for index, (stage, key, prediction) in enumerate([("young", "u", primary_u), ("mature", "v", primary_v)]):
        residual = oof[f"{key}1_log"].to_numpy(float) - oof[prediction].to_numpy(float)
        standardized_column = f"standardized_residual_{key}"
        standardized = oof[standardized_column].to_numpy(float) if standardized_column in oof else (residual - np.mean(residual)) / max(np.std(residual, ddof=1), 1e-12)
        finite = standardized[np.isfinite(standardized)]
        if len(finite) >= 8:
            normal = normaltest(finite)
            normal_statistic, normal_p = float(normal.statistic), float(normal.pvalue)
        else:
            normal_statistic, normal_p = np.nan, np.nan
        serial, pairs = residual_serial_statistic(oof.assign(**{standardized_column: standardized}), standardized_column)
        serial_p = residual_serial_permutation_p(oof.assign(**{standardized_column: standardized}), standardized_column, serial, config.residual_permutations, config.seed + 88000 + index)
        expected_count = oof[f"point_mean_{key}_kha"].to_numpy(float) * forecast_sampling_area(oof, key) / 10.0
        rows.append({
            "stage": stage,
            "residual_mean": float(np.mean(residual)),
            "residual_sd": float(np.std(residual, ddof=1)),
            "standardized_mean": float(np.mean(finite)),
            "standardized_sd": float(np.std(finite, ddof=1)),
            "standardized_skew": float(pd.Series(finite).skew()),
            "standardized_excess_kurtosis": float(pd.Series(finite).kurt()),
            "tail_abs_gt_2": float(np.mean(np.abs(finite) > 2)),
            "tail_abs_gt_3": float(np.mean(np.abs(finite) > 3)),
            "normaltest_statistic": normal_statistic,
            "normaltest_p": normal_p,
            "spearman_residual_fitted": safe_spearman(residual, oof[prediction].to_numpy(float)),
            "spearman_abs_residual_dt": safe_spearman(np.abs(residual), oof.dt_years.to_numpy(float)),
            "spearman_squared_standardized_dt": safe_spearman(standardized * standardized, oof.dt_years.to_numpy(float)),
            "spearman_squared_standardized_expected_count": safe_spearman(standardized * standardized, expected_count),
            "within_plot_lag1_correlation": serial,
            "within_plot_lag_pairs": pairs,
            "within_plot_lag1_permutation_p": serial_p,
        })
    return pd.DataFrame(rows)

def derivative_summary(oof_derivatives: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    if oof_derivatives.empty:
        return pd.DataFrame()
    rows = []
    for feature, frame in oof_derivatives.groupby("feature"):
        for stage, column, hessian in [("young", "gradient_young", "hessian_diag_young"), ("mature", "gradient_mature", "hessian_diag_mature")]:
            values = frame[column].to_numpy(float)
            def statistic(sample: pd.DataFrame, column=column) -> float:
                return float(np.mean(sample[column].to_numpy(float)))
            distribution = bootstrap_statistic(frame, statistic, config.cluster_bootstrap, config.seed + 99000 + sum(ord(c) for c in feature + stage))
            low, high = bootstrap_interval(distribution)
            rows.append({
                "feature": feature,
                "stage": stage,
                "mean_gradient": float(np.mean(values)),
                "median_gradient": float(np.median(values)),
                "mean_absolute_gradient": float(np.mean(np.abs(values))),
                "gradient_q05": float(np.quantile(values, 0.05)),
                "gradient_q95": float(np.quantile(values, 0.95)),
                "fraction_positive_gradient": float(np.mean(values > 0)),
                "mean_gradient_site_bootstrap_CI_low": low,
                "mean_gradient_site_bootstrap_CI_high": high,
                "mean_hessian_diagonal": float(frame[hessian].mean()),
                "mean_absolute_hessian_diagonal": float(frame[hessian].abs().mean()),
            })
    return pd.DataFrame(rows).sort_values(["stage", "mean_absolute_gradient"], ascending=[True, False]).reset_index(drop=True)

def climate_rate_response(scaffold: DemographicScaffold, label: str, grid_points: int = 101) -> pd.DataFrame:
    grid = np.linspace(-2.5, 2.5, grid_points)
    rows = []
    for varied in [0, 1]:
        climate = np.zeros((len(grid), 2))
        climate[:, varied] = grid
        q, maturation, mortality, rho, mu, a, b = DemographicScaffold.rates(scaffold.parameters, climate)
        for index, value in enumerate(grid):
            rows.append({
                "fit": label,
                "varied_component": f"PC{varied + 1}",
                "component_value": value,
                "recruitment_q": float(q[index]),
                "maturation_F": float(maturation[index]),
                "mature_loss_H": float(mortality[index]),
                "rho": float(rho),
                "mu": float(mu),
                "young_density_regulation_a": float(a),
                "mature_density_regulation_b": float(b),
            })
    return pd.DataFrame(rows)

def summarize_climate_rates(fold_rates: pd.DataFrame) -> pd.DataFrame:
    if fold_rates.empty:
        return pd.DataFrame()
    rows = []
    for (component, value), frame in fold_rates.groupby(["varied_component", "component_value"]):
        for rate in ["recruitment_q", "maturation_F", "mature_loss_H"]:
            rows.append({
                "varied_component": component,
                "component_value": value,
                "rate": rate,
                "median": float(frame[rate].median()),
                "lower": float(frame[rate].quantile(0.025)),
                "upper": float(frame[rate].quantile(0.975)),
                "minimum": float(frame[rate].min()),
                "maximum": float(frame[rate].max()),
                "n_fits": len(frame),
            })
    return pd.DataFrame(rows)

def parameter_stability(fold_parameters: pd.DataFrame) -> pd.DataFrame:
    if fold_parameters.empty:
        return pd.DataFrame()
    rows = []
    for parameter, frame in fold_parameters.groupby("parameter"):
        values = frame.estimate_natural.to_numpy(float)
        internal = frame.estimate_internal.to_numpy(float)
        rows.append({
            "parameter": parameter,
            "n_fits": len(frame),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "relative_range": float((np.max(values) - np.min(values)) / (abs(np.median(values)) + 1e-8)),
            "active_bound_fraction": float(frame.active_bound.mean()),
            "sign_consistency_internal": float(max(np.mean(internal >= 0), np.mean(internal <= 0))),
        })
    return pd.DataFrame(rows)

def ode_profile_objective(scaffold: DemographicScaffold, data: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    if scaffold is None or config.ode_profile_points < 3:
        return pd.DataFrame()
    rng = np.random.default_rng(config.seed + 9110000)
    if len(data) > config.ode_profile_rows:
        index = np.sort(rng.choice(len(data), size=config.ode_profile_rows, replace=False))
        sample = data.iloc[index].reset_index(drop=True)
    else:
        sample = data.reset_index(drop=True)
    climate = scaffold.climate_scores(sample)
    profile_weight_u, profile_weight_v, _ = cross_fitted_estimation_weights(sample, config, config.seed + 9110100)
    young_weight = np.sqrt(np.maximum(profile_weight_u, 1e-12))
    mature_weight = np.sqrt(np.maximum(profile_weight_v, 1e-12))
    young_scale = np.std(sample.delta_u_log.to_numpy(float)) + 0.10
    mature_scale = np.std(sample.delta_v_log.to_numpy(float)) + 0.05
    lower = np.array([-8, -3, -3, -8, -3, -3, -8, -3, -3, -6, -6, -8, -8], dtype=float)
    upper = np.array([4, 3, 3, 4, 3, 3, 4, 3, 3, 1, 1, 1, 1], dtype=float)
    names = ["q_intercept", "q_PC1", "q_PC2", "F_intercept", "F_PC1", "F_PC2", "H_intercept", "H_PC1", "H_PC2", "log_rho", "log_mu", "log_a", "log_b"]
    profile_indices = [0, 3, 6, 9, 10, 11, 12]
    reference_initial = np.array([logit(np.clip((0.08 - Q_MIN) / (Q_MAX - Q_MIN), 1e-5, 1 - 1e-5)), 0.0, 0.0, logit(np.clip((0.08 - F_MIN) / (F_MAX - F_MIN), 1e-5, 1 - 1e-5)), 0.0, 0.0, logit(np.clip((0.06 - H_MIN) / (H_MAX - H_MIN), 1e-5, 1 - 1e-5)), 0.0, 0.0, np.log(0.03), np.log(0.04), np.log(0.02), np.log(0.05)])
    climate_indices = np.array([1, 2, 4, 5, 7, 8], dtype=int)

    def residual(parameters: np.ndarray) -> np.ndarray:
        pred_u, pred_v = DemographicScaffold.integrate_arrays(sample, climate, parameters, max(8, config.integration_steps - 2))
        data_residual = np.concatenate([(pred_u - sample.u1_log.to_numpy(float)) * young_weight / young_scale, (pred_v - sample.v1_log.to_numpy(float)) * mature_weight / mature_scale])
        penalty_scale = 0.08 if scaffold.use_climate else 2.0
        climate_penalty = penalty_scale * parameters[climate_indices]
        positive_parameter_penalty = 0.05 * (parameters[9:13] - reference_initial[9:13])
        return np.concatenate([data_residual, climate_penalty, positive_parameter_penalty])

    rows = []
    reference = scaffold.parameters.copy()
    for parameter_index in profile_indices:
        half_width = max(0.35, 0.15 * (upper[parameter_index] - lower[parameter_index]))
        grid = np.linspace(max(lower[parameter_index] + 1e-5, reference[parameter_index] - half_width), min(upper[parameter_index] - 1e-5, reference[parameter_index] + half_width), config.ode_profile_points)
        free = np.array([i for i in range(len(reference)) if i != parameter_index], dtype=int)
        start_free = reference[free].copy()
        parameter_rows = []
        for grid_index, fixed_value in enumerate(grid):
            def objective_free(free_values: np.ndarray) -> np.ndarray:
                parameters = reference.copy()
                parameters[parameter_index] = fixed_value
                parameters[free] = free_values
                return residual(parameters)
            result = least_squares(objective_free, start_free, bounds=(lower[free], upper[free]), loss="soft_l1", f_scale=0.70, max_nfev=config.ode_profile_max_nfev, xtol=1e-6, ftol=1e-6, gtol=1e-6)
            if result.success and np.all(np.isfinite(result.x)):
                start_free = result.x.copy()
            parameter_rows.append({"parameter": names[parameter_index], "grid_index": grid_index, "fixed_value_internal": float(fixed_value), "profile_cost": float(result.cost), "fit_success": bool(result.success), "nfev": int(result.nfev), "rows_used": len(sample)})
        minimum = min(row["profile_cost"] for row in parameter_rows)
        for row in parameter_rows:
            row["delta_profile_cost"] = float(row["profile_cost"] - minimum)
            rows.append(row)
    return pd.DataFrame(rows)

def ode_identifiability(scaffold: DemographicScaffold, data: pd.DataFrame, config: Configuration, stochastic: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    empty = {"summary": pd.DataFrame(), "singular_values": pd.DataFrame(), "weighted_singular_values": pd.DataFrame(), "parameter_correlation": pd.DataFrame(), "weighted_parameter_correlation": pd.DataFrame(), "sensitivity": pd.DataFrame(), "profile_objective": pd.DataFrame()}
    if scaffold is None:
        return empty
    rng = np.random.default_rng(config.seed + 9100000)
    if len(data) > config.ode_sensitivity_rows:
        index = np.sort(rng.choice(len(data), size=config.ode_sensitivity_rows, replace=False))
        sample = data.iloc[index].reset_index(drop=True)
    else:
        sample = data.reset_index(drop=True)
    climate = scaffold.climate_scores(sample)
    _, _, jac_u, jac_v = DemographicScaffold.integrate_arrays_with_jacobian(sample, climate, scaffold.parameters, scaffold.integration_steps)
    matrix = np.vstack([jac_u, jac_v])
    p = matrix.shape[1]
    names = ["q_intercept", "q_PC1", "q_PC2", "F_intercept", "F_PC1", "F_PC2", "H_intercept", "H_PC1", "H_PC2", "log_rho", "log_mu", "log_a", "log_b"]
    sensitivity_rows = []
    for j, name in enumerate(names):
        column = matrix[:, j]
        sensitivity_rows.append({"parameter": name, "mean_abs_endpoint_sensitivity": float(np.mean(np.abs(column))), "max_abs_endpoint_sensitivity": float(np.max(np.abs(column))), "rms_endpoint_sensitivity": float(np.sqrt(np.mean(column ** 2)))})

    def decomposition(values: np.ndarray) -> tuple[np.ndarray, float, float, float, int, pd.DataFrame]:
        singular = np.linalg.svd(values, full_matrices=False, compute_uv=False)
        largest = float(singular[0]) if len(singular) else np.nan
        smallest = float(singular[-1]) if len(singular) else np.nan
        condition = float(largest / smallest) if np.isfinite(smallest) and smallest > 1e-14 else float("inf")
        tolerance = largest * 1e-6 if np.isfinite(largest) else np.nan
        effective_rank = int(np.sum(singular > tolerance)) if np.isfinite(tolerance) else 0
        gram = values.T @ values
        covariance = np.linalg.pinv(gram, rcond=1e-10)
        diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        correlation = covariance / np.maximum(diagonal[:, None] * diagonal[None, :], 1e-15)
        correlation = np.clip(correlation, -1.0, 1.0)
        correlation_rows = []
        for i, left in enumerate(names):
            for j, right in enumerate(names):
                correlation_rows.append({"parameter_1": left, "parameter_2": right, "approximate_correlation": float(correlation[i, j])})
        return singular, largest, smallest, condition, effective_rank, pd.DataFrame(correlation_rows)

    singular, largest, smallest, condition, effective_rank, correlation = decomposition(matrix)
    if stochastic is not None:
        reference_components = persistence_reference_components(sample)
        variance_u, variance_v = predictive_variance_from_components(sample, reference_components, stochastic, config)
        weighted_matrix = np.vstack([jac_u / np.sqrt(np.maximum(variance_u, 1e-10))[:, None], jac_v / np.sqrt(np.maximum(variance_v, 1e-10))[:, None]])
    else:
        weighted_matrix = matrix.copy()
    weighted_singular, weighted_largest, weighted_smallest, weighted_condition, weighted_rank, weighted_correlation = decomposition(weighted_matrix)
    singular_table = pd.DataFrame({"index": np.arange(1, len(singular) + 1), "singular_value": singular, "relative_to_largest": singular / max(largest, 1e-15)})
    weighted_singular_table = pd.DataFrame({"index": np.arange(1, len(weighted_singular) + 1), "singular_value": weighted_singular, "relative_to_largest": weighted_singular / max(weighted_largest, 1e-15)})
    summary = pd.DataFrame([{"rows_used": len(sample), "parameters": p, "largest_singular_value": largest, "smallest_singular_value": smallest, "condition_number": condition, "effective_rank_1e_minus_6": effective_rank, "full_rank": effective_rank == p, "weighted_largest_singular_value": weighted_largest, "weighted_smallest_singular_value": weighted_smallest, "weighted_condition_number": weighted_condition, "weighted_effective_rank_1e_minus_6": weighted_rank, "weighted_full_rank": weighted_rank == p, "weighted_by_predictive_variance": stochastic is not None, "jacobian_method": "analytic_discrete_Heun_sensitivity"}])
    profile = ode_profile_objective(scaffold, data, config)
    return {"summary": summary, "singular_values": singular_table, "weighted_singular_values": weighted_singular_table, "parameter_correlation": correlation, "weighted_parameter_correlation": weighted_correlation, "sensitivity": pd.DataFrame(sensitivity_rows), "profile_objective": profile}

def climate_rate_response_raw(scaffold: DemographicScaffold, reference_data: pd.DataFrame, label: str, grid_points: int = 21) -> pd.DataFrame:
    if scaffold is None:
        return pd.DataFrame()
    medians = reference_data[RAW_CLIMATE_MODEL].median(numeric_only=True)
    rows = []
    for feature in RAW_CLIMATE_MODEL:
        values = np.linspace(float(reference_data[feature].quantile(0.05)), float(reference_data[feature].quantile(0.95)), grid_points)
        frame = pd.DataFrame({name: np.full(len(values), float(medians[name])) for name in RAW_CLIMATE_MODEL})
        frame[feature] = values
        climate_scores = scaffold.climate.transform(frame) if scaffold.climate is not None else np.zeros((len(frame), 2))
        q, maturation, mortality, rho, mu, a, b = DemographicScaffold.rates(scaffold.parameters, climate_scores)
        for index, value in enumerate(values):
            rows.append({"fit": label, "varied_feature": feature, "feature_value": float(value), "recruitment_q": float(q[index]), "maturation_F": float(maturation[index]), "mature_loss_H": float(mortality[index]), "rho": float(rho), "mu": float(mu), "young_density_regulation_a": float(a), "mature_density_regulation_b": float(b)})
    return pd.DataFrame(rows)

def site_bootstrap_scaffold(data: pd.DataFrame, reference_scaffold: DemographicScaffold, config: Configuration) -> dict[str, pd.DataFrame]:
    if config.ode_bootstrap <= 0:
        return {"draws": pd.DataFrame(), "summary": pd.DataFrame(), "rates": pd.DataFrame(), "rate_summary": pd.DataFrame()}
    rng = np.random.default_rng(config.seed + 9200000)
    sites = data.siteID.astype(str).unique()
    parameter_draws = []
    rate_draws = []
    names = ["q_intercept", "q_PC1", "q_PC2", "F_intercept", "F_PC1", "F_PC2", "H_intercept", "H_PC1", "H_PC2", "log_rho", "log_mu", "log_a", "log_b"]
    for bootstrap in range(1, config.ode_bootstrap + 1):
        sampled_sites = rng.choice(sites, size=len(sites), replace=True)
        pieces = []
        for draw_index, site in enumerate(sampled_sites):
            piece = data[data.siteID.astype(str) == site].copy()
            piece["siteID"] = piece.siteID.astype(str) + f"__bootstrap_{draw_index}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            weight_u, weight_v, _ = cross_fitted_estimation_weights(sample, config, config.seed + 9250000 + bootstrap * 41)
            fitted = DemographicScaffold.fit(sample, config, config.seed + 9200000 + bootstrap * 37, reference_scaffold.use_climate, False, climate_override=None, multistart_override=config.scaffold_bootstrap_multistart, max_nfev_override=config.scaffold_bootstrap_max_nfev, sample_weight_young=weight_u, sample_weight_mature=weight_v)
        except Exception:
            continue
        for index, (name, estimate) in enumerate(zip(names, fitted.parameters)):
            parameter_draws.append({"bootstrap": bootstrap, "parameter": name, "estimate_internal": float(estimate), "estimate_natural": float(np.exp(estimate)) if index >= 9 else float(estimate), "active_bound": int(fitted.active_bounds[index]), "fit_success": fitted.fit_success, "climate_basis_refit": True})
        rates = climate_rate_response_raw(fitted, data, f"bootstrap_{bootstrap}", grid_points=21)
        rates.insert(0, "bootstrap", bootstrap)
        rate_draws.append(rates)
    draws = pd.DataFrame(parameter_draws)
    summaries = []
    if not draws.empty:
        for parameter, frame in draws.groupby("parameter"):
            internal = frame.estimate_internal.to_numpy(float)
            natural = frame.estimate_natural.to_numpy(float)
            basis_specific = parameter in {"q_PC1", "q_PC2", "F_PC1", "F_PC2", "H_PC1", "H_PC2"}
            summaries.append({"parameter": parameter, "bootstrap_successes": int(frame.bootstrap.nunique()), "estimate_internal_median": float(np.median(internal)), "estimate_internal_CI_low": float(np.quantile(internal, 0.025)), "estimate_internal_CI_high": float(np.quantile(internal, 0.975)), "estimate_natural_median": float(np.median(natural)), "estimate_natural_CI_low": float(np.quantile(natural, 0.025)), "estimate_natural_CI_high": float(np.quantile(natural, 0.975)), "probability_internal_positive": float(np.mean(internal > 0)), "active_bound_fraction": float(frame.active_bound.mean()), "climate_basis_refit_each_bootstrap": True, "PC_coefficient_basis_specific": basis_specific})
    rates = pd.concat(rate_draws, ignore_index=True) if rate_draws else pd.DataFrame()
    rate_summary_rows = []
    if not rates.empty:
        for (feature, value), frame in rates.groupby(["varied_feature", "feature_value"]):
            for rate in ["recruitment_q", "maturation_F", "mature_loss_H"]:
                values = frame[rate].to_numpy(float)
                rate_summary_rows.append({"varied_feature": feature, "feature_value": value, "rate": rate, "median": float(np.median(values)), "CI_low": float(np.quantile(values, 0.025)), "CI_high": float(np.quantile(values, 0.975)), "bootstrap_successes": int(frame.bootstrap.nunique())})
    return {"draws": draws, "summary": pd.DataFrame(summaries), "rates": rates, "rate_summary": pd.DataFrame(rate_summary_rows)}

def derivative_bootstrap_stability(data: pd.DataFrame, spec: ModelSpec, hyper: HyperParameters, config: Configuration) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.derivative_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(config.seed + 9300000)
    sites = data.siteID.astype(str).unique()
    if len(data) > config.derivative_reference_rows:
        reference_index = np.sort(rng.choice(len(data), size=config.derivative_reference_rows, replace=False))
        reference = data.iloc[reference_index].reset_index(drop=True)
    else:
        reference = data.reset_index(drop=True)
    rows = []
    for bootstrap in range(1, config.derivative_bootstrap + 1):
        sampled_sites = rng.choice(sites, size=len(sites), replace=True)
        sample = pd.concat([data[data.siteID.astype(str) == site] for site in sampled_sites], ignore_index=True)
        try:
            model = TRACEModel.fit(sample, spec, hyper, config, config.seed + 9300000 + bootstrap * 97)
            derivative = model.discrepancy.derivatives(reference, raw_scale=True)
        except Exception:
            continue
        for feature_index, feature in enumerate(model.discrepancy.transform.feature_names):
            rows.append({"bootstrap": bootstrap, "feature": feature, "stage": "young", "mean_gradient": float(np.mean(derivative["gradient_u"][:, feature_index])), "mean_absolute_gradient": float(np.mean(np.abs(derivative["gradient_u"][:, feature_index]))), "mean_hessian_diagonal": float(np.mean(derivative["hessian_diag_u"][:, feature_index]))})
            rows.append({"bootstrap": bootstrap, "feature": feature, "stage": "mature", "mean_gradient": float(np.mean(derivative["gradient_v"][:, feature_index])), "mean_absolute_gradient": float(np.mean(np.abs(derivative["gradient_v"][:, feature_index]))), "mean_hessian_diagonal": float(np.mean(derivative["hessian_diag_v"][:, feature_index]))})
    draws = pd.DataFrame(rows)
    summaries = []
    if not draws.empty:
        for (feature, stage), frame in draws.groupby(["feature", "stage"]):
            values = frame.mean_gradient.to_numpy(float)
            summaries.append({
                "feature": feature,
                "stage": stage,
                "bootstrap_successes": int(frame.bootstrap.nunique()),
                "mean_gradient_median": float(np.median(values)),
                "mean_gradient_CI_low": float(np.quantile(values, 0.025)),
                "mean_gradient_CI_high": float(np.quantile(values, 0.975)),
                "probability_gradient_positive": float(np.mean(values > 0)),
                "mean_absolute_gradient_median": float(frame.mean_absolute_gradient.median()),
                "mean_hessian_diagonal_median": float(frame.mean_hessian_diagonal.median()),
                "hyperparameters_fixed_across_bootstrap": True,
            })
    return draws, pd.DataFrame(summaries)

def surrogate_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = ["u0_log", "v0_log", "log_dt", "target_year", "decimalLatitude", "decimalLongitude", "young_baseline_area_precision", "mature_baseline_area_precision", "young_baseline_area_missing", "mature_baseline_area_missing", "zero_u0", "zero_v0"]
    x = frame[names].to_numpy(float)
    x[:, 3] -= np.mean(x[:, 3])
    return x, names

def fit_sparse_discrepancy_surrogate(oof: pd.DataFrame, config: Configuration) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, names = surrogate_matrix(oof)
    groups = oof.siteID.astype(str).to_numpy()
    splits = list(GroupKFold(min(3, np.unique(groups).size)).split(x, groups=groups))
    polynomial = PolynomialFeatures(2, include_bias=False)
    polynomial.fit(np.zeros((1, x.shape[1])))
    terms = polynomial.get_feature_names_out(names)
    grid = [(alpha, l1) for alpha in (np.logspace(-4, -1, 4) if config.profile == "smoke" else np.logspace(-4, -1, 7)) for l1 in ((1.0,) if config.profile == "smoke" else (0.8, 1.0))]

    def imputer(train_matrix: np.ndarray) -> np.ndarray:
        values = np.zeros(train_matrix.shape[1], dtype=float)
        for j in range(train_matrix.shape[1]):
            finite = train_matrix[:, j][np.isfinite(train_matrix[:, j])]
            values[j] = float(np.median(finite)) if finite.size else 0.0
        return values

    def apply_imputer(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(matrix), matrix, values[None, :])

    term_rows = []
    audit = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        y = oof[f"discrepancy_{key}_log"].to_numpy(float)
        scores = []
        for alpha, l1 in grid:
            fold_scores = []
            for train, valid in splits:
                fill = imputer(x[train])
                train_x = apply_imputer(x[train], fill)
                valid_x = apply_imputer(x[valid], fill)
                scaler1 = StandardScaler().fit(train_x)
                train_poly = polynomial.transform(scaler1.transform(train_x))
                scaler2 = StandardScaler().fit(train_poly)
                model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=config.surrogate_max_iter, random_state=config.seed).fit(scaler2.transform(train_poly), y[train])
                valid_poly = polynomial.transform(scaler1.transform(valid_x))
                fold_scores.append(mean_squared_error(y[valid], model.predict(scaler2.transform(valid_poly))))
            scores.append((float(np.mean(fold_scores)), alpha, l1))
        _, alpha, l1 = min(scores)
        oof_prediction = np.zeros(len(y))
        for train, valid in splits:
            fill = imputer(x[train])
            train_x = apply_imputer(x[train], fill)
            valid_x = apply_imputer(x[valid], fill)
            scaler1 = StandardScaler().fit(train_x)
            train_poly = polynomial.transform(scaler1.transform(train_x))
            scaler2 = StandardScaler().fit(train_poly)
            model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=config.surrogate_max_iter, random_state=config.seed).fit(scaler2.transform(train_poly), y[train])
            oof_prediction[valid] = model.predict(scaler2.transform(polynomial.transform(scaler1.transform(valid_x))))
        fill = imputer(x)
        all_x = apply_imputer(x, fill)
        scaler1 = StandardScaler().fit(all_x)
        all_poly = polynomial.transform(scaler1.transform(all_x))
        scaler2 = StandardScaler().fit(all_poly)
        final = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=config.surrogate_max_iter, random_state=config.seed).fit(scaler2.transform(all_poly), y)
        coefficients = final.coef_
        selected = np.zeros(len(coefficients))
        positive = np.zeros(len(coefficients))
        negative = np.zeros(len(coefficients))
        splitter = GroupShuffleSplit(config.surrogate_splits, test_size=0.25, random_state=config.seed + 5000)
        for train, _ in splitter.split(x, groups=groups):
            fill = imputer(x[train])
            train_x = apply_imputer(x[train], fill)
            a = StandardScaler().fit(train_x)
            p = polynomial.transform(a.transform(train_x))
            b = StandardScaler().fit(p)
            model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=config.surrogate_max_iter, random_state=config.seed).fit(b.transform(p), y[train])
            c = model.coef_
            nonzero = np.abs(c) > 1e-9
            selected += nonzero
            positive += c > 1e-9
            negative += c < -1e-9
        frequency = selected / max(config.surrogate_splits, 1)
        sign = np.maximum(positive, negative) / np.maximum(selected, 1)
        table = pd.DataFrame({"stage": stage, "term": terms, "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients), "selection_frequency": frequency, "sign_stability": sign, "claimable": (frequency >= 0.70) & (sign >= 0.80) & (np.abs(coefficients) > 1e-9)}).sort_values(["claimable", "absolute_coefficient"], ascending=[False, False]).head(config.surrogate_top_terms)
        term_rows.append(table)
        audit.append({"stage": stage, "grouped_OOF_R2": float(r2_score(y, oof_prediction)), "grouped_OOF_RMSE": float(np.sqrt(mean_squared_error(y, oof_prediction))), "alpha": alpha, "l1_ratio": l1, "nonzero_terms": int(np.sum(np.abs(coefficients) > 1e-9)), "claimable_terms": int(np.sum((frequency >= 0.70) & (sign >= 0.80) & (np.abs(coefficients) > 1e-9))), "candidate_terms": len(coefficients), "fold_specific_missing_value_imputation": True})
    return pd.concat(term_rows, ignore_index=True), pd.DataFrame(audit)

def run_repeated_holdouts(data: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    if config.repeated_holdouts <= 0:
        return pd.DataFrame()
    rows = []
    groups = data.siteID.astype(str).to_numpy()
    splitter = GroupShuffleSplit(config.repeated_holdouts, test_size=config.repeated_test_fraction, random_state=config.seed + 4000000)
    for repeat, (train_index, test_index) in enumerate(splitter.split(np.arange(len(data)), groups=groups), 1):
        train = data.iloc[train_index].reset_index(drop=True)
        test = data.iloc[test_index].reset_index(drop=True)
        core_index, calibration_index = split_core_calibration(train, config.calibration_fraction, config.seed + repeat * 123)
        core = train.iloc[core_index].reset_index(drop=True)
        calibration = train.iloc[calibration_index].reset_index(drop=True)
        spec = primary_spec()
        hyper, _ = tune_trace_model(core, spec, config, config.seed + 5000000 + repeat * 100003)
        model = TRACEModel.fit(core, spec, hyper, config, config.seed + 5100000 + repeat * 100003)
        stochastic, stochastic_audit = fit_stochastic_layer_cross_fitted(core, spec, hyper, config, config.seed + 5150000 + repeat * 100003)
        state = fit_calibration_state(model, calibration, stochastic, core.siteID.astype(str).unique().tolist(), stochastic_audit, config, config.seed + 5200000 + repeat * 100003)
        prediction, _ = predict_with_calibration(model, test, state, config, config.seed + 5300000 + repeat * 100003)
        for stage, key in [("young", "u"), ("mature", "v")]:
            y = test[f"{key}1_log"].to_numpy(float)
            point = prediction[f"pred_{key}_log"].to_numpy(float)
            persistence = test[f"{key}0_log"].to_numpy(float)
            rmse = float(np.sqrt(mean_squared_error(y, point)))
            reference = float(np.sqrt(mean_squared_error(y, persistence)))
            covered = (y >= prediction[f"lower_{key}_log"]) & (y <= prediction[f"upper_{key}_log"])
            site = pd.DataFrame({"siteID": test.siteID.astype(str), "covered": covered}).groupby("siteID").covered.agg(["mean", "all"])
            rows.append({
                "repeat": repeat,
                "stage": stage,
                "train_sites": train.siteID.nunique(),
                "core_sites": core.siteID.nunique(),
                "calibration_sites": calibration.siteID.nunique(),
                "test_sites": test.siteID.nunique(),
                "selected_gate": hyper.gate,
                "selected_ridge_scale": hyper.ridge_scale,
                "selected_distillation_weight": hyper.distillation_weight,
                "selected_occurrence_gate": hyper.occurrence_gate,
                "selected_gamma_scale": hyper.gamma_scale,
                "selected_kernel_alpha": hyper.kernel_alpha,
                "RMSE_log1p": rmse,
                "Persistence_RMSE_log1p": reference,
                "RMSE_skill": 1.0 - rmse / reference if reference > 1e-12 else np.nan,
                "transition_coverage": float(np.mean(covered)),
                "mean_site_coverage": float(site["mean"].mean()),
                "simultaneous_site_coverage": float(site["all"].mean()),
                "mean_interval_width": float(np.mean(prediction[f"upper_{key}_log"] - prediction[f"lower_{key}_log"])),
            })
    return pd.DataFrame(rows)

def evaluate_train_test_split(train: pd.DataFrame, test: pd.DataFrame, split_label: str, split_type: str, config: Configuration, seed: int) -> pd.DataFrame:
    if train.siteID.nunique() < config.minimum_temporal_train_sites or test.siteID.nunique() < config.minimum_temporal_test_sites or len(test) < 20:
        return pd.DataFrame()
    core_index, calibration_index = split_core_calibration(train, config.calibration_fraction, seed + 101)
    core = train.iloc[core_index].reset_index(drop=True)
    calibration = train.iloc[calibration_index].reset_index(drop=True)
    if core.siteID.nunique() < 4 or calibration.siteID.nunique() < 2:
        return pd.DataFrame()
    rows = []
    specifications = [primary_spec(), ModelSpec("TRACE without realized climate", use_climate=False)]
    for model_index, spec in enumerate(specifications):
        model_seed = seed + model_index * 100003
        hyper, _ = tune_trace_model(core, spec, config, model_seed + 1000)
        model = TRACEModel.fit(core, spec, hyper, config, model_seed + 2000)
        stochastic, stochastic_audit = fit_stochastic_layer_cross_fitted(core, spec, hyper, config, model_seed + 3000)
        state = fit_calibration_state(model, calibration, stochastic, core.siteID.astype(str).unique().tolist(), stochastic_audit, config, model_seed + 4000)
        prediction, _ = predict_with_calibration(model, test, state, config, model_seed + 5000)
        for stage, key in [("young", "u"), ("mature", "v")]:
            y = test[f"{key}1_log"].to_numpy(float)
            point = prediction[f"pred_{key}_log"].to_numpy(float)
            persistence = test[f"{key}0_log"].to_numpy(float)
            rmse = float(np.sqrt(mean_squared_error(y, point)))
            reference = float(np.sqrt(mean_squared_error(y, persistence)))
            covered = (y >= prediction[f"lower_{key}_log"].to_numpy(float)) & (y <= prediction[f"upper_{key}_log"].to_numpy(float))
            site_coverage = pd.DataFrame({"siteID": test.siteID.astype(str), "covered": covered}).groupby("siteID").covered.agg(["mean", "all"])
            rows.append({"split_type": split_type, "split_label": split_label, "model": spec.name, "stage": stage, "train_rows": len(train), "test_rows": len(test), "train_sites": int(train.siteID.nunique()), "core_sites": int(core.siteID.nunique()), "calibration_sites": int(calibration.siteID.nunique()), "test_sites": int(test.siteID.nunique()), "selected_gate": hyper.gate, "selected_ridge_scale": hyper.ridge_scale, "selected_distillation_weight": hyper.distillation_weight, "selected_occurrence_gate": hyper.occurrence_gate, "selected_gamma_scale": hyper.gamma_scale, "selected_kernel_alpha": hyper.kernel_alpha, "RMSE_log1p": rmse, "Persistence_RMSE_log1p": reference, "RMSE_skill": 1.0 - rmse / reference if reference > 1e-12 else np.nan, "MAE_log1p": float(mean_absolute_error(y, point)), "Spearman": safe_spearman(y, point), "transition_coverage": float(np.mean(covered)), "mean_site_coverage": float(site_coverage["mean"].mean()), "simultaneous_site_coverage": float(site_coverage["all"].mean()), "mean_interval_width": float(np.mean(prediction[f"upper_{key}_log"].to_numpy(float) - prediction[f"lower_{key}_log"].to_numpy(float))), "stochastic_calibration_site_overlap": len(set(state.stochastic_source_sites) & set(state.calibration_sites))})
    return pd.DataFrame(rows)

def run_temporal_validation(data: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    if not config.temporal_validation_enabled or "t0_date" not in data or "t1_date" not in data or not data.t0_date.notna().any() or not data.t1_date.notna().any():
        return pd.DataFrame()
    valid_dates = data.t1_date.dropna().astype("int64").to_numpy()
    rows = []
    for index, quantile in enumerate(config.temporal_cut_quantiles):
        cutoff_ns = int(np.quantile(valid_dates, quantile))
        cutoff = pd.to_datetime(cutoff_ns).normalize()
        train = data[data.t1_date < cutoff].reset_index(drop=True)
        test = data[data.t0_date >= cutoff].reset_index(drop=True)
        result = evaluate_train_test_split(train, test, cutoff.strftime("%Y-%m-%d"), "forward_temporal", config, config.seed + 6100000 + index * 100003)
        if not result.empty:
            result["cutoff"] = cutoff
            result["site_overlap_train_test"] = len(set(train.siteID.astype(str)) & set(test.siteID.astype(str)))
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def run_spatiotemporal_validation(data: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    if not config.temporal_validation_enabled or config.spatiotemporal_repeats <= 0 or "t0_date" not in data or "t1_date" not in data or not data.t0_date.notna().any() or not data.t1_date.notna().any():
        return pd.DataFrame()
    valid_dates = data.t1_date.dropna().astype("int64").to_numpy()
    cutoff = pd.to_datetime(int(np.quantile(valid_dates, 0.65))).normalize()
    groups = data.siteID.astype(str).to_numpy()
    splitter = GroupShuffleSplit(config.spatiotemporal_repeats, test_size=config.spatiotemporal_test_fraction, random_state=config.seed + 6200000)
    rows = []
    for repeat, (train_index, held_index) in enumerate(splitter.split(np.arange(len(data)), groups=groups), 1):
        train_sites = set(data.iloc[train_index].siteID.astype(str))
        held_sites = set(data.iloc[held_index].siteID.astype(str))
        train = data[data.siteID.astype(str).isin(train_sites) & (data.t1_date < cutoff)].reset_index(drop=True)
        test = data[data.siteID.astype(str).isin(held_sites) & (data.t0_date >= cutoff)].reset_index(drop=True)
        if set(train.siteID.astype(str)) & set(test.siteID.astype(str)):
            raise RuntimeError("Spatiotemporal site overlap detected")
        result = evaluate_train_test_split(train, test, f"repeat_{repeat}", "site_excluded_future_period", config, config.seed + 6300000 + repeat * 100003)
        if not result.empty:
            result["cutoff"] = cutoff
            result["repeat"] = repeat
            result["site_overlap_train_test"] = 0
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def fit_final_deployment(data: pd.DataFrame, config: Configuration) -> tuple[TRACEModel, CalibrationState, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_index, calibration_index = split_core_calibration(data, config.calibration_fraction, config.seed + 777)
    core = data.iloc[core_index].reset_index(drop=True)
    calibration = data.iloc[calibration_index].reset_index(drop=True)
    spec = primary_spec()
    hyper, tuning = tune_trace_model(core, spec, config, config.seed + 8000000)
    model = TRACEModel.fit(core, spec, hyper, config, config.seed + 8100000)
    stochastic, stochastic_audit = fit_stochastic_layer_cross_fitted(core, spec, hyper, config, config.seed + 8150000)
    state = fit_calibration_state(model, calibration, stochastic, core.siteID.astype(str).unique().tolist(), stochastic_audit, config, config.seed + 8200000)
    prediction, _ = predict_with_calibration(model, data.reset_index(drop=True), state, config, config.seed + 8300000)
    identity = data[["row_id", "transition_id", "siteID", "plotID", "dt_years", "u0_kha", "v0_kha", "u1_kha", "v1_kha"]].reset_index(drop=True)
    final = pd.concat([identity, prediction], axis=1)
    for key in ["u", "v"]:
        for prefix in ["pred", "lower", "upper", "scaffold", "positive_magnitude"]:
            column = f"{prefix}_{key}_log"
            if column in final:
                final[f"{prefix}_{key}_kha"] = np.expm1(final[column])
    split = pd.DataFrame({
        "partition": ["core", "calibration"],
        "rows": [len(core), len(calibration)],
        "sites": [core.siteID.nunique(), calibration.siteID.nunique()],
        "site_ids": [json.dumps(sorted(core.siteID.astype(str).unique().tolist())), json.dumps(sorted(calibration.siteID.astype(str).unique().tolist()))],
    })
    return model, state, final, split, tuning

def numerical_audit(model: TRACEModel, data: pd.DataFrame, cv_audit: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    components = model.components(data)
    h = model.hyper.occurrence_gate if model.occurrence is not None else 0.0
    expected_u = (1.0 - h) * components["magnitude_u"] + h * components["hurdle_original_log_u"]
    expected_v = (1.0 - h) * components["magnitude_v"] + h * components["hurdle_original_log_v"]
    point_coherence_u = float(np.max(np.abs(components["pred_u"] - expected_u)))
    point_coherence_v = float(np.max(np.abs(components["pred_v"] - expected_v)))
    if model.scaffold is not None:
        coarse_u, coarse_v = model.scaffold.predict(data, steps=config.integration_steps)
        fine_u, fine_v = model.scaffold.predict(data, steps=config.integration_steps * 2)
        integration_difference = float(max(np.max(np.abs(coarse_u - fine_u)), np.max(np.abs(coarse_v - fine_v))))
    else:
        integration_difference = 0.0
    derivatives = model.discrepancy.derivatives(data.iloc[: min(len(data), 256)].reset_index(drop=True), raw_scale=True)
    derivative_finite = all(np.isfinite(value).all() for value in derivatives.values())
    rows = [
        {"item": "all_outer_site_overlaps_zero", "value": bool((cv_audit[["train_test_site_overlap", "core_calibration_site_overlap", "core_test_site_overlap", "calibration_test_site_overlap"]].to_numpy() == 0).all()) if not cv_audit.empty else False},
        {"item": "point_estimand_coherence_max_error_young", "value": point_coherence_u},
        {"item": "point_estimand_coherence_max_error_mature", "value": point_coherence_v},
        {"item": "double_integration_resolution_max_difference", "value": integration_difference},
        {"item": "all_point_predictions_finite", "value": bool(np.isfinite(components["pred_u"]).all() and np.isfinite(components["pred_v"]).all())},
        {"item": "all_point_predictions_nonnegative", "value": bool((components["pred_u"] >= 0).all() and (components["pred_v"] >= 0).all())},
        {"item": "all_analytic_derivatives_finite", "value": derivative_finite},
        {"item": "kernel_centers", "value": int(len(model.discrepancy.centers))},
        {"item": "selected_gate", "value": model.hyper.gate},
        {"item": "selected_ridge_scale", "value": model.hyper.ridge_scale},
        {"item": "selected_distillation_weight", "value": model.hyper.distillation_weight},
        {"item": "selected_occurrence_gate", "value": model.hyper.occurrence_gate},
        {"item": "selected_gamma_scale", "value": model.hyper.gamma_scale},
        {"item": "selected_kernel_alpha", "value": model.hyper.kernel_alpha},
        {"item": "variance_weight_crossfit_overlap_zero", "value": bool((model.estimation_weight_audit.get("site_overlap", pd.Series([0])).fillna(0).to_numpy(float) == 0).all()) if not model.estimation_weight_audit.empty else True},
    ]
    return pd.DataFrame(rows)

def contribution_evidence(oof: pd.DataFrame, metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("scaffold", "Discrepancy only, independently retuned", "removal"),
        ("teacher", "Without teacher smoothing", "removal"),
        ("realized_climate", "Without realized climate", "removal"),
        ("coordinates", "Without coordinates", "removal"),
        ("variance_weighting", "Without variance-aware estimation", "removal"),
        ("sampling_precision_predictors", "Without sampling-precision predictors", "removal"),
        ("sampling_support", "Without sampling support information", "removal"),
        ("occurrence", "Without occurrence layer", "removal"),
    ]
    rows = []
    for component, comparison, comparison_type in comparisons:
        for stage in ["young", "mature"]:
            primary = metrics[(metrics.model == primary_spec().name) & (metrics.stage == stage) & (metrics.metric == "RMSE_log1p")]
            alternative = metrics[(metrics.model == comparison) & (metrics.stage == stage) & (metrics.metric == "RMSE_log1p")]
            if primary.empty or alternative.empty:
                continue
            primary_rmse = float(primary.estimate.iloc[0])
            alternative_rmse = float(alternative.estimate.iloc[0])
            paired_row = paired[(paired.stage == stage) & (paired.comparison_model == comparison)]
            comparison_minus_primary = alternative_rmse - primary_rmse
            if comparison_type == "removal":
                component_gain = comparison_minus_primary
                probability_supported = float(paired_row.probability_primary_better.iloc[0]) if not paired_row.empty else np.nan
                ci_low = float(paired_row.CI_low.iloc[0]) if not paired_row.empty else np.nan
                ci_high = float(paired_row.CI_high.iloc[0]) if not paired_row.empty else np.nan
            else:
                component_gain = -comparison_minus_primary
                probability_supported = float(1.0 - paired_row.probability_primary_better.iloc[0]) if not paired_row.empty else np.nan
                ci_low = -float(paired_row.CI_high.iloc[0]) if not paired_row.empty else np.nan
                ci_high = -float(paired_row.CI_low.iloc[0]) if not paired_row.empty else np.nan
            rows.append({
                "component": component,
                "stage": stage,
                "comparison_type": comparison_type,
                "primary_RMSE": primary_rmse,
                "comparison_RMSE": alternative_rmse,
                "component_RMSE_gain": component_gain,
                "RMSE_gain_comparison_minus_primary": comparison_minus_primary,
                "relative_component_gain": component_gain / primary_rmse if primary_rmse > 0 else np.nan,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "bootstrap_probability_component_supported": probability_supported,
                "bootstrap_probability_primary_better": float(paired_row.probability_primary_better.iloc[0]) if not paired_row.empty else np.nan,
                "site_equal_MSE_signflip_p": float(paired_row.site_equal_MSE_signflip_p.iloc[0]) if not paired_row.empty else np.nan,
                "comparison_independently_refit_and_retuned": True,
            })
    for stage, key in [("young", "u"), ("mature", "v")]:
        rows.append({
            "component": "magnitude_decomposition",
            "stage": stage,
            "comparison_type": "diagnostic",
            "primary_RMSE": np.nan,
            "comparison_RMSE": np.nan,
            "component_RMSE_gain": float(np.mean(np.abs(oof[f"gated_scaffold_{key}_log"]))),
            "RMSE_gain_comparison_minus_primary": float(np.mean(np.abs(oof[f"gated_scaffold_{key}_log"]))),
            "relative_component_gain": float(np.mean(np.abs(oof[f"discrepancy_{key}_log"]))),
            "CI_low": np.nan,
            "CI_high": np.nan,
            "bootstrap_probability_component_supported": np.nan,
            "bootstrap_probability_primary_better": np.nan,
            "site_equal_MSE_signflip_p": np.nan,
            "comparison_independently_refit_and_retuned": True,
        })
    return pd.DataFrame(rows)

def research_gap_evidence(metrics: pd.DataFrame, contribution: pd.DataFrame, regimes: pd.DataFrame, calibration: pd.DataFrame, repeated: pd.DataFrame, distribution: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ["young", "mature"]:
        full = metrics[(metrics.model == primary_spec().name) & (metrics.stage == stage) & (metrics.metric == "RMSE_log1p")]
        persistence = metrics[(metrics.model == "Persistence") & (metrics.stage == stage) & (metrics.metric == "RMSE_log1p")]
        if not full.empty and not persistence.empty:
            skill = 1.0 - float(full.estimate.iloc[0]) / float(persistence.estimate.iloc[0])
            rows.append({"question": "unseen-site transition prediction", "stage": stage, "estimate": skill, "status": "supported" if skill > 0.05 else "weak", "statement": f"Site-excluded RMSE skill over persistence was {skill:.1%}."})
        for component in ["scaffold", "teacher", "realized_climate", "coordinates", "variance_weighting", "sampling_precision_predictors", "sampling_support", "occurrence"]:
            row = contribution[(contribution.component == component) & (contribution.stage == stage)]
            if row.empty:
                continue
            gain = float(row.component_RMSE_gain.iloc[0])
            probability = float(row.bootstrap_probability_component_supported.iloc[0])
            status = "supported" if gain > 0 and probability >= 0.95 else "not_supported"
            comparison_type = str(row.comparison_type.iloc[0])
            rows.append({"question": f"incremental predictive value of {component}", "stage": stage, "estimate": gain, "status": status, "statement": f"The independently refit and retuned {comparison_type} comparison implied an RMSE gain of {gain:.4f}; bootstrap support probability was {probability:.3f}."})
        regime = regimes[(regimes.stage == stage) & (regimes.regime == "zero_to_positive")]
        if not regime.empty:
            skill = float(regime.skill_vs_persistence.iloc[0])
            low = float(regime.skill_CI_low.iloc[0])
            high = float(regime.skill_CI_high.iloc[0])
            rows.append({"question": "zero-to-positive transfer", "stage": stage, "estimate": skill, "status": "supported" if np.isfinite(low) and low > 0 else "uncertain", "statement": f"Zero-to-positive skill was {skill:.1%} with site-bootstrap interval [{low:.1%}, {high:.1%}]."})
        calibrated = calibration[(calibration.stage == stage) & (calibration.interval == "site_conformal")]
        if not calibrated.empty:
            coverage = float(calibrated.transition_coverage.iloc[0])
            simultaneous = float(calibrated.simultaneous_site_coverage.iloc[0])
            rows.append({"question": "site-aware predictive uncertainty", "stage": stage, "estimate": coverage, "status": "transition_scale_only" if simultaneous < 0.8 else "broad", "statement": f"Transition coverage was {coverage:.1%}; simultaneous whole-site coverage was {simultaneous:.1%}."})
        endpoint = distribution[(distribution.stage == stage) & (distribution.distribution == "distribution")]
        for sensitivity_name, question in [("log_diffusion", "embedded log-diffusion sensitivity"), ("positive_diffusion", "positive-state demographic/multiplicative diffusion sensitivity")]:
            sensitivity = distribution[(distribution.stage == stage) & (distribution.distribution == sensitivity_name)]
            if not sensitivity.empty and not endpoint.empty:
                difference = float(endpoint.mean_CRPS_log.iloc[0] - sensitivity.mean_CRPS_log.iloc[0])
                rows.append({"question": question, "stage": stage, "estimate": difference, "status": "sensitivity_distribution_better" if difference > 0 else "endpoint_noise_better_or_equal", "statement": f"{question} changed mean CRPS by {-difference:.4f} relative to endpoint-noise simulation; this is a sensitivity comparison, not an estimated SDE claim."})
        diagnostic = residual[residual.stage == stage]
        if not diagnostic.empty:
            p = float(diagnostic.within_plot_lag1_permutation_p.iloc[0])
            rows.append({"question": "white-noise residual assumption", "stage": stage, "estimate": float(diagnostic.within_plot_lag1_correlation.iloc[0]), "status": "serial_dependence_detected" if p < 0.05 else "no_strong_serial_evidence", "statement": f"Within-plot lag-1 standardized residual correlation had permutation p={p:.4f}."})
    if not repeated.empty:
        for stage, frame in repeated.groupby("stage"):
            rows.append({"question": "fully nested repeated holdout stability", "stage": stage, "estimate": float(frame.RMSE_skill.median()), "status": "supported" if float(np.mean(frame.RMSE_skill > 0)) >= 0.8 else "heterogeneous", "statement": f"Median fully nested repeated-holdout skill was {frame.RMSE_skill.median():.1%}; {int(np.sum(frame.RMSE_skill > 0))}/{len(frame)} holdouts were positive."})
    return pd.DataFrame(rows)

def paper_claims(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=["claim"])
    return pd.DataFrame({"claim": evidence.statement.astype(str)})

def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = data.select_dtypes(include=[np.number])
    for column in numeric.columns:
        values = numeric[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        rows.append({
            "variable": column,
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
            "skew": float(pd.Series(values).skew()),
            "excess_kurtosis": float(pd.Series(values).kurt()),
        })
    return pd.DataFrame(rows)

def site_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, frame in data.groupby("siteID"):
        rows.append({
            "siteID": str(site),
            "transitions": len(frame),
            "plots": frame.plotID.nunique(),
            "dt_mean": float(frame.dt_years.mean()),
            "dt_min": float(frame.dt_years.min()),
            "dt_max": float(frame.dt_years.max()),
            "young_zero_origin_fraction": float(frame.zero_u0.mean()),
            "mature_zero_origin_fraction": float(frame.zero_v0.mean()),
            "young_zero_endpoint_fraction": float(1.0 - frame.positive_u1.mean()),
            "mature_zero_endpoint_fraction": float(1.0 - frame.positive_v1.mean()),
        })
    return pd.DataFrame(rows)

def zero_transition_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        start_positive = data[f"{key}0_kha"].to_numpy(float) > 0
        end_positive = data[f"{key}1_kha"].to_numpy(float) > 0
        regimes = np.select([~start_positive & ~end_positive, ~start_positive & end_positive, start_positive & ~end_positive, start_positive & end_positive], ["zero_to_zero", "zero_to_positive", "positive_to_zero", "positive_to_positive"], default="unknown")
        for regime in np.unique(regimes):
            mask = regimes == regime
            rows.append({"stage": stage, "regime": regime, "transitions": int(mask.sum()), "sites": int(data.loc[mask, "siteID"].nunique()), "fraction": float(np.mean(mask))})
    return pd.DataFrame(rows)

def build_summary(data: pd.DataFrame, metrics: pd.DataFrame, site_equal: pd.DataFrame, runtime: float) -> pd.DataFrame:
    rows = [
        {"metric": "transitions", "value": len(data)},
        {"metric": "plots", "value": data.plotID.nunique()},
        {"metric": "sites", "value": data.siteID.nunique()},
        {"metric": "runtime_seconds", "value": runtime},
    ]
    for stage in ["young", "mature"]:
        full = metrics[(metrics.model == primary_spec().name) & (metrics.stage == stage) & (metrics.metric == "RMSE_log1p")]
        persistence = metrics[(metrics.model == "Persistence") & (metrics.stage == stage) & (metrics.metric == "RMSE_log1p")]
        if not full.empty and not persistence.empty:
            rmse = float(full.estimate.iloc[0])
            reference = float(persistence.estimate.iloc[0])
            rows.extend([
                {"metric": f"{stage}_TRACE_RMSE_log1p", "value": rmse},
                {"metric": f"{stage}_persistence_RMSE_log1p", "value": reference},
                {"metric": f"{stage}_RMSE_skill_vs_persistence", "value": 1.0 - rmse / reference},
            ])
        equal = site_equal[(site_equal.model == primary_spec().name) & (site_equal.stage == stage)]
        if not equal.empty:
            rows.append({"metric": f"{stage}_site_equal_skill_vs_persistence", "value": float(equal.site_equal_skill_vs_persistence.iloc[0])})
    return pd.DataFrame(rows)

def model_card(config: Configuration) -> pd.DataFrame:
    statements = [
        ("Scope", "Primary next-census prediction uses baseline state, known forecast horizon, coordinates and reconstructed baseline sampling support under complete-site exclusion; realized interval climate is evaluated only as a secondary conditional extension."),
        ("Point estimand", "The point estimator interpolates between conditional positive-state log magnitude and the log transform of the original-scale hurdle mean through a grouped-CV-selected occurrence gate h."),
        ("Distribution estimand", "The predictive distribution retains the full calibrated hurdle representation; its log-scale expectation p(x)m(x) is kept distinct from the RMSE-oriented point estimator."),
        ("Demographic scaffold", "A non-negative two-state ODE provides a constrained structural reference; its parameters are treated as descriptive rather than causal demographic rates."),
        ("Discrepancy", "A sparse multi-scale Gaussian RBF residual model is fitted directly to the gated scaffold residual and exposes exact gradients and Hessian diagnostics."),
        ("Hyperparameter selection", "Mechanistic gate, ridge scale, teacher distillation weight, occurrence-to-point gate, RBF gamma scale and two-kernel mixture weight are selected by prespecified staged grouped inner cross-validation within each development split."),
        ("Feature geometry", "Standardized predictor dimensions receive neutral unit weights; kernel mixture weights are selected rather than hand assigned."),
        ("Estimation weighting", "Primary mean-model estimation uses cross-fitted feasible inverse-variance weights based on the same process-plus-expected-count variance form, with a persistence reference and site-excluded weight construction."),
        ("Ablations", "Every TRACE comparison is independently refitted and the applicable prespecified tuning dimensions are reselected inside each outer training fold."),
        ("Occurrence", "Site-grouped cross-fitted Extra Trees probabilities are calibrated by isotonic regression or Platt scaling and are evaluated with discrimination, proper probability scores and calibration diagnostics."),
        ("Uncertainty", "Endpoint-noise Monte Carlo distributions combine process-scale variation, expected-count observation variation, cross-stratum dependence and occurrence uncertainty, then receive site-aware split-conformal adjustment."),
        ("SDE sensitivity", "Log-state and positive-state stochastic scaffold variants are retained as sensitivity models and are evaluated using proper distributional scores rather than treated as identified stochastic dynamics."),
        ("Inference", "Site bootstrap intervals are used for predictive contrasts and transition regimes; ODE parameters receive site-bootstrap uncertainty and Jacobian-SVD identifiability diagnostics."),
        ("Residual diagnostics", "Standardized residual distribution, heteroskedasticity, PIT behavior and within-plot lag dependence are audited instead of assuming white Gaussian noise without testing."),
        ("Calibration target", f"The configured conformal target is {config.conformal_mode}; transition-level calibration is not interpreted as simultaneous whole-site coverage unless site_max is used and empirically supported."),
        ("Not claimed", "The framework does not identify causal climate effects, does not establish unrestricted spatial extrapolation and does not validate recursive long-horizon dynamics from endpoint data alone."),
    ]
    return pd.DataFrame(statements, columns=["item", "statement"])

def make_figures(
    oof: pd.DataFrame,
    regimes: pd.DataFrame,
    calibration: pd.DataFrame,
    derivatives: pd.DataFrame,
    residual: pd.DataFrame,
    singular_values: pd.DataFrame,
    distribution: pd.DataFrame,
    out: Path,
    config: Configuration,
) -> pd.DataFrame:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    primary_u, primary_v = prediction_columns_for_model(primary_spec().name)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, stage, target, prediction in [(axes[0], "Young", "u1_log", primary_u), (axes[1], "Mature", "v1_log", primary_v)]:
        x = oof[target].to_numpy(float)
        y = oof[prediction].to_numpy(float)
        axis.hexbin(x, y, gridsize=45, mincnt=1)
        limit = max(float(np.max(x)), float(np.max(y)))
        axis.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
        axis.set_xlabel("Observed log(1+density)")
        axis.set_ylabel("Predicted log(1+density)")
        axis.set_title(stage)
    fig.tight_layout()
    path = figure_dir / "observed_predicted.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": "observed_predicted", "path": str(path)})
    if not regimes.empty:
        display = regimes[regimes.regime != "zero_to_zero"].copy()
        fig, axis = plt.subplots(figsize=(9, 4.5))
        order = ["positive_to_positive", "positive_to_zero", "zero_to_positive"]
        x = np.arange(len(order))
        width = 0.36
        for offset, stage in [(-width / 2, "young"), (width / 2, "mature")]:
            frame = display[display.stage == stage].set_index("regime").reindex(order)
            values = frame.skill_vs_persistence.to_numpy(float) * 100
            low = (frame.skill_vs_persistence - frame.skill_CI_low).to_numpy(float) * 100
            high = (frame.skill_CI_high - frame.skill_vs_persistence).to_numpy(float) * 100
            axis.bar(x + offset, values, width, label=stage)
            axis.errorbar(x + offset, values, yerr=np.vstack([np.maximum(low, 0), np.maximum(high, 0)]), fmt="none", capsize=3)
        axis.axhline(0, linestyle="--", linewidth=1)
        axis.set_xticks(x, [value.replace("_", " ") for value in order], rotation=20)
        axis.set_ylabel("RMSE skill vs persistence (%)")
        axis.legend()
        fig.tight_layout()
        path = figure_dir / "regime_skill_ci.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": "regime_skill_ci", "path": str(path)})
    if not calibration.empty:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        labels = [f"{row.stage}\n{row.interval}" for row in calibration.itertuples()]
        values = calibration.transition_coverage.to_numpy(float)
        axis.bar(np.arange(len(values)), values)
        axis.axhline(config.interval, linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Transition coverage")
        fig.tight_layout()
        path = figure_dir / "coverage.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": "coverage", "path": str(path)})
    if not derivatives.empty:
        top = derivatives.sort_values("mean_absolute_gradient", ascending=False).groupby("stage").head(8)
        fig, axis = plt.subplots(figsize=(10, 5))
        labels = [f"{row.stage}: {row.feature}" for row in top.itertuples()]
        axis.barh(np.arange(len(top)), top.mean_absolute_gradient.to_numpy(float))
        axis.set_yticks(np.arange(len(top)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Mean absolute analytic gradient")
        fig.tight_layout()
        path = figure_dir / "discrepancy_gradient_sensitivity.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": "discrepancy_gradient_sensitivity", "path": str(path)})
    if "standardized_residual_u" in oof and "standardized_residual_v" in oof:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, stage, column in [(axes[0], "Young", "standardized_residual_u"), (axes[1], "Mature", "standardized_residual_v")]:
            values = np.sort(oof[column].to_numpy(float))
            probabilities = (np.arange(1, len(values) + 1) - 0.5) / len(values)
            theoretical = scipy.stats.norm.ppf(probabilities)
            axis.scatter(theoretical, values, s=7)
            lo = min(theoretical.min(), values.min())
            hi = max(theoretical.max(), values.max())
            axis.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            axis.set_xlabel("Normal quantile")
            axis.set_ylabel("Standardized residual quantile")
            axis.set_title(stage)
        fig.tight_layout()
        path = figure_dir / "residual_qq.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": "residual_qq", "path": str(path)})
    if not singular_values.empty:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.semilogy(singular_values["index"], singular_values["relative_to_largest"], marker="o")
        axis.set_xlabel("Sensitivity singular-value index")
        axis.set_ylabel("Relative singular value")
        fig.tight_layout()
        path = figure_dir / "ode_identifiability_svd.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": "ode_identifiability_svd", "path": str(path)})
    if not distribution.empty and distribution.distribution.nunique() > 1:
        pivot = distribution.pivot(index="stage", columns="distribution", values="mean_CRPS_log")
        fig, axis = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(pivot.index))
        columns = list(pivot.columns)
        width = 0.8 / len(columns)
        for j, column in enumerate(columns):
            axis.bar(x - 0.4 + width / 2 + j * width, pivot[column].to_numpy(float), width, label=column)
        axis.set_xticks(x, pivot.index)
        axis.set_ylabel("Mean CRPS on log scale")
        axis.legend()
        fig.tight_layout()
        path = figure_dir / "stochastic_sensitivity_crps_comparison.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": "stochastic_sensitivity_crps_comparison", "path": str(path)})
    return pd.DataFrame(manifest)

def safe_sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[\\/*?:\[\]]", "_", name)[:31] or "Sheet"
    candidate = base
    counter = 1
    while candidate in used:
        suffix = f"_{counter}"
        candidate = base[: 31 - len(suffix)] + suffix
        counter += 1
    used.add(candidate)
    return candidate

def save_excel_workbook(out: Path, tables: dict[str, pd.DataFrame], cell_limit: int = 150000) -> Path:
    path = out / "trace_results.xlsx"
    used = set()
    large = []
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
            if frame.shape[0] * max(frame.shape[1], 1) > cell_limit:
                large.append({"table": name, "rows": len(frame), "columns": frame.shape[1], "csv": f"tables/{slug(name)}.csv"})
                continue
            sheet = safe_sheet_name(name, used)
            frame.to_excel(writer, sheet_name=sheet, index=False)
        if large:
            pd.DataFrame(large).to_excel(writer, sheet_name=safe_sheet_name("Large_Tables_Index", used), index=False)
    return path

def save_tables(out: Path, tables: dict[str, pd.DataFrame]) -> Path:
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        frame.to_csv(table_dir / f"{slug(name)}.csv", index=False)
    return save_excel_workbook(out, tables)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default="results_trace_final")
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="full")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--paired-bootstrap", type=int, default=None)
    parser.add_argument("--regime-bootstrap", type=int, default=None)
    parser.add_argument("--ode-bootstrap", type=int, default=None)
    parser.add_argument("--derivative-bootstrap", type=int, default=None)
    parser.add_argument("--mc-draws", type=int, default=None)
    parser.add_argument("--log-diffusion-draws", type=int, default=None)
    parser.add_argument("--positive-diffusion-draws", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-diffusion-steps", type=int, default=None)
    parser.add_argument("--positive-diffusion-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=None)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--conformal-mode", choices=["site_max", "site_quantile", "transition"], default=None)
    parser.add_argument("--repeated-holdouts", type=int, default=None)
    parser.add_argument("--rbf-centers", type=int, default=None)
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument("--no-benchmark-tuning", action="store_true")
    parser.add_argument("--no-log-diffusion", action="store_true")
    parser.add_argument("--no-positive-diffusion", action="store_true")
    parser.add_argument("--no-temporal-validation", action="store_true")
    parser.add_argument("--no-ode-bootstrap", action="store_true")
    parser.add_argument("--no-derivative-bootstrap", action="store_true")
    args, _ = parser.parse_known_args()
    return args

def resolve_data_path(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input data file does not exist: {candidate}")
        return candidate
    directories = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    preferred = ["data(7).csv", "data.csv", "dataset.csv"]
    candidates = []
    for directory in directories:
        candidates.extend(directory / name for name in preferred if (directory / name).exists())
        candidates.extend(sorted(directory.glob("*.csv")))
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            columns = set(pd.read_csv(resolved, nrows=0).columns)
        except Exception:
            continue
        if set(REQUIRED).issubset(columns):
            return resolved
    raise FileNotFoundError("No compatible CSV was found. Use --data PATH.")

def main() -> None:
    args = parse_arguments()
    start = time.time()
    data_path = resolve_data_path(args.data)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = Configuration.for_profile(
        args.profile,
        outer_folds=args.folds,
        cluster_bootstrap=args.bootstrap,
        paired_bootstrap=args.paired_bootstrap,
        regime_bootstrap=args.regime_bootstrap,
        ode_bootstrap=args.ode_bootstrap,
        derivative_bootstrap=args.derivative_bootstrap,
        mc_draws=args.mc_draws,
        log_diffusion_draws=args.log_diffusion_draws,
        positive_diffusion_draws=args.positive_diffusion_draws,
        integration_steps=args.steps,
        log_diffusion_steps=args.log_diffusion_steps,
        positive_diffusion_steps=args.positive_diffusion_steps,
        seed=args.seed,
        dpi=args.dpi,
        calibration_fraction=args.calibration_fraction,
        interval=args.interval,
        conformal_mode=args.conformal_mode,
        repeated_holdouts=args.repeated_holdouts,
        rbf_centers=args.rbf_centers,
    )
    if args.no_tuning:
        config.tune_model = False
    if args.no_benchmark_tuning:
        config.tune_benchmarks = False
    if args.no_log_diffusion:
        config.log_diffusion_enabled = False
    if args.no_positive_diffusion:
        config.positive_diffusion_enabled = False
    if args.no_temporal_validation:
        config.temporal_validation_enabled = False
    if args.no_ode_bootstrap:
        config.ode_bootstrap = 0
    if args.no_derivative_bootstrap:
        config.derivative_bootstrap = 0
    data = prepare_data(data_path)
    print(f"{len(data)} transitions | {data.plotID.nunique()} plots | {data.siteID.nunique()} sites | profile={config.profile}")
    evaluation = run_outer_cv(data, config)
    oof = evaluation["OOF_Predictions"]
    metrics = model_metrics_table(oof, config)
    paired = paired_model_comparisons(oof, config)
    folds = fold_metrics(oof)
    sites = grouped_skill(oof, "siteID")
    site_equal = site_equal_metrics(oof)
    horizon = horizon_skill(oof)
    ood = ood_analysis(oof)
    regimes = transition_regime_analysis(oof, config)
    occurrence = occurrence_metrics(oof)
    reliability = occurrence_reliability(oof)
    occurrence_calibration = occurrence_calibration_summary(oof)
    support_sensitivity = zero_origin_support_sensitivity(oof, config)
    calibration = interval_calibration_diagnostics(oof, config)
    distribution = distribution_diagnostics(oof, config)
    pit_bins = pit_cluster_bins(oof, config)
    residual = residual_diagnostics(oof, config)
    derivative = derivative_summary(evaluation["OOF_Derivatives"], config)
    contribution = contribution_evidence(oof, metrics, paired)
    climate = summarize_climate_rates(evaluation["Fold_Climate_Rates"])
    stability = parameter_stability(evaluation["Fold_Parameters"])
    surrogate_terms, surrogate_audit = fit_sparse_discrepancy_surrogate(oof, config)
    repeated = run_repeated_holdouts(data, config)
    temporal = run_temporal_validation(data, config)
    spatiotemporal = run_spatiotemporal_validation(data, config)
    final_model, final_calibration, final_prediction, final_split, final_tuning = fit_final_deployment(data, config)
    identifiability = ode_identifiability(final_model.scaffold, data, config, final_calibration.stochastic)
    ode_bootstrap = site_bootstrap_scaffold(data, final_model.scaffold, config) if final_model.scaffold is not None else {"draws": pd.DataFrame(), "summary": pd.DataFrame(), "rates": pd.DataFrame(), "rate_summary": pd.DataFrame()}
    derivative_bootstrap_draws, derivative_bootstrap_summary = derivative_bootstrap_stability(data, primary_spec(), final_model.hyper, config)
    numerical = numerical_audit(final_model, data, evaluation["CV_Audit"], config)
    evidence = research_gap_evidence(metrics, contribution, regimes, calibration, repeated, distribution, residual)
    claims = paper_claims(evidence)
    runtime = time.time() - start
    summary = build_summary(data, metrics, site_equal, runtime)
    descriptive = descriptive_statistics(data[[column for column in REQUIRED if column in data.columns] + ["target_year", "u0_log", "v0_log", "u1_log", "v1_log", "delta_u_log", "delta_v_log", "log_dt"]])
    site_descriptive = site_descriptive_statistics(data)
    zero_descriptive = zero_transition_statistics(data)
    data_dictionary = data_role_dictionary(data)
    predictor_audit = predictor_availability_audit(data, final_model.spec, final_model.discrepancy.transform.feature_names)
    data_quality = data_quality_audit(data)
    missingness = missingness_audit(data)
    final_parameters = final_model.scaffold.parameter_table("final_deployment") if final_model.scaffold is not None else pd.DataFrame()
    final_rates = climate_rate_response(final_model.scaffold, "final_deployment") if final_model.scaffold is not None else pd.DataFrame()
    final_hessian = final_model.discrepancy.mean_abs_hessian_pairs(data.iloc[: min(len(data), config.derivative_reference_rows)].reset_index(drop=True), raw_scale=True)
    code_path = Path(__file__).resolve()
    temporal_status = {"requested": bool(config.temporal_validation_enabled), "completed": bool(not temporal.empty), "rows": int(len(temporal)), "reason": None if not temporal.empty else "no_eligible_split_or_insufficient_sites"}
    spatiotemporal_status = {"requested": bool(config.temporal_validation_enabled and config.spatiotemporal_repeats > 0), "completed": bool(not spatiotemporal.empty), "rows": int(len(spatiotemporal)), "reason": None if not spatiotemporal.empty else "no_eligible_site_excluded_future_split_or_insufficient_sites"}
    run_audit = pd.DataFrame({
        "item": ["model", "data", "data_sha256", "code", "code_sha256", "python", "numpy", "pandas", "scipy", "scikit_learn", "platform", "configuration", "final_spec", "final_hyperparameters", "teacher_occurrence_audit", "calibration", "forward_temporal_validation", "spatiotemporal_validation"],
        "value": [
            MODEL_NAME,
            str(data_path),
            sha256(data_path),
            str(code_path),
            sha256(code_path),
            sys.version,
            np.__version__,
            pd.__version__,
            scipy.__version__,
            sklearn.__version__,
            platform.platform(),
            json.dumps(jsonable(asdict(config)), ensure_ascii=False),
            json.dumps(jsonable(asdict(final_model.spec)), ensure_ascii=False),
            json.dumps(jsonable(asdict(final_model.hyper)), ensure_ascii=False),
            json.dumps(jsonable(final_model.teacher_audit), ensure_ascii=False),
            json.dumps({"mode": final_calibration.mode, "q_young": final_calibration.conformal_q_young, "q_mature": final_calibration.conformal_q_mature, "calibration_sites": final_calibration.calibration_sites, "stochastic_source_sites": final_calibration.stochastic_source_sites}, ensure_ascii=False),
            json.dumps(temporal_status, ensure_ascii=False),
            json.dumps(spatiotemporal_status, ensure_ascii=False),
        ],
    })
    tables = {
        "Summary": summary,
        "Paper_Claims": claims,
        "Research_Evidence": evidence,
        "Model_Metrics": metrics,
        "Paired_Model_Comparisons": paired,
        "Component_Evidence": contribution,
        "Transition_Regimes": regimes,
        "Repeated_Holdouts": repeated,
        "Forward_Temporal_Validation": temporal,
        "Spatiotemporal_Validation": spatiotemporal,
        "Fold_Metrics": folds,
        "Site_Skill": sites,
        "Site_Equal_Metrics": site_equal,
        "Horizon_Skill": horizon,
        "OOD_Analysis": ood,
        "Interval_Calibration": calibration,
        "Distribution_Diagnostics": distribution,
        "PIT_Cluster_Bins": pit_bins,
        "Residual_Diagnostics": residual,
        "Occurrence_Metrics": occurrence,
        "Occurrence_Reliability": reliability,
        "Occurrence_Calibration_Summary": occurrence_calibration,
        "Zero_Origin_Support_Sensitivity": support_sensitivity,
        "TRACE_Tuning": evaluation["TRACE_Tuning"],
        "Benchmark_Tuning": evaluation["Benchmark_Tuning"],
        "Model_Selections": evaluation["Model_Selections"],
        "CV_Audit": evaluation["CV_Audit"],
        "Stochastic_Fit_Audit": evaluation["Stochastic_Fit_Audit"],
        "Estimation_Weight_Audit": evaluation["Estimation_Weight_Audit"],
        "Parameter_Stability": stability,
        "Climate_Rate_Stability": climate,
        "ODE_Identifiability": identifiability["summary"],
        "ODE_Singular_Values": identifiability["singular_values"],
        "ODE_Weighted_Singular_Values": identifiability["weighted_singular_values"],
        "ODE_Parameter_Correlation": identifiability["parameter_correlation"],
        "ODE_Weighted_Parameter_Correlation": identifiability["weighted_parameter_correlation"],
        "ODE_Parameter_Sensitivity": identifiability["sensitivity"],
        "ODE_Profile_Objective": identifiability["profile_objective"],
        "ODE_Bootstrap_Draws": ode_bootstrap["draws"],
        "ODE_Bootstrap_Summary": ode_bootstrap["summary"],
        "ODE_Bootstrap_Rates": ode_bootstrap["rates"],
        "ODE_Bootstrap_Rate_Summary": ode_bootstrap["rate_summary"],
        "OOF_Derivatives": evaluation["OOF_Derivatives"],
        "Derivative_Summary": derivative,
        "OOF_Hessian_Interactions": evaluation["OOF_Hessian_Interactions"],
        "Final_Hessian_Interactions": final_hessian,
        "Derivative_Bootstrap_Draws": derivative_bootstrap_draws,
        "Derivative_Bootstrap_Summary": derivative_bootstrap_summary,
        "Discrepancy_Surrogate_Terms": surrogate_terms,
        "Discrepancy_Surrogate_Audit": surrogate_audit,
        "Descriptive_Statistics": descriptive,
        "Site_Descriptive": site_descriptive,
        "Zero_Transition_Statistics": zero_descriptive,
        "Data_Dictionary": data_dictionary,
        "Predictor_Availability_Audit": predictor_audit,
        "Data_Quality_Audit": data_quality,
        "Missingness_Audit": missingness,
        "Final_Parameters": final_parameters,
        "Final_Climate_Rates": final_rates,
        "Final_Split": final_split,
        "Final_Tuning": final_tuning,
        "Final_Calibration_Scores": final_calibration.score_table,
        "Final_Stochastic_Fit_Audit": final_calibration.stochastic_audit,
        "Final_Estimation_Weight_Audit": final_model.estimation_weight_audit,
        "OOF_Predictions": oof,
        "Final_Predictions": final_prediction,
        "Numerical_Audit": numerical,
        "Model_Card": model_card(config),
        "Run_Audit": run_audit,
    }
    figures = make_figures(oof, regimes, calibration, derivative, residual, identifiability["singular_values"], distribution, out, config)
    tables["Figures_Manifest"] = figures
    workbook = save_tables(out, tables)
    model_payload = {
        "model_name": MODEL_NAME,
        "configuration": asdict(config),
        "model": final_model,
        "calibration": final_calibration,
        "data_sha256": sha256(data_path),
        "code_sha256": sha256(code_path),
    }
    model_path = out / "trace_model.joblib"
    paper_results_path = out / "paper_results.txt"
    joblib.dump(model_payload, model_path, compress=3)
    paper_results_path.write_text("\n".join(claims.claim.astype(str)), encoding="utf-8")
    manifest = {
        "model": MODEL_NAME,
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "code": str(code_path),
        "code_sha256": sha256(code_path),
        "workbook_sha256": sha256(workbook),
        "model_sha256": sha256(model_path),
        "paper_results_sha256": sha256(paper_results_path),
        "configuration": jsonable(asdict(config)),
        "runtime_seconds": runtime,
        "validation_status": {"forward_temporal": temporal_status, "spatiotemporal": spatiotemporal_status},
        "guarantees": {
            "site_blocked_outer_test": True,
            "calibration_sites_excluded_from_predictor_fit": True,
            "same_predictor_for_calibration_and_test": True,
            "applicable_prespecified_tuning_dimensions_reselected_for_each_trace_comparison": True,
            "point_estimator_occurrence_adjustment_selected_inside_grouped_inner_cv": True,
            "predictive_distribution_kept_distinct_from_rmse_oriented_point_estimator": True,
            "inverse_variance_estimation_weights_cross_fitted_by_site": bool(final_model.spec.use_variance_weights),
            "primary_model_uses_realized_interval_climate": bool(final_model.spec.use_climate),
            "realized_climate_evaluated_as_primary_covariate": bool(final_model.spec.use_climate),
            "no_realized_climate_ablation_completed": "pred_u_log__without_realized_climate" in oof.columns and "pred_v_log__without_realized_climate" in oof.columns,
            "endpoint_year_excluded_from_predictor_feature_set": "event_year" not in final_model.discrepancy.transform.feature_names,
            "direct_residual_fit_to_gated_scaffold": True,
            "analytic_discrepancy_gradients": True,
            "analytic_discrepancy_hessian_diagnostics": True,
            "site_bootstrap_predictive_inference": True,
            "ode_site_bootstrap_refits_climate_basis": config.ode_bootstrap > 0,
            "ode_jacobian_svd_identifiability": True,
            "ode_weighted_identifiability_uses_baseline_available_count_support": True,
            "ode_profile_objective_diagnostics": config.ode_profile_points >= 3,
            "fully_nested_repeated_holdouts_completed": bool(config.repeated_holdouts > 0 and not repeated.empty),
            "stochastic_parameters_estimated_without_conformal_labels": True,
            "candidate_gate_residual_refit_without_algebraic_shortcut": True,
            "cluster_aware_PIT_diagnostics_without_IID_KS_pvalue": True,
            "forward_temporal_validation_completed": bool(not temporal.empty),
            "spatiotemporal_site_excluded_validation_completed": bool(not spatiotemporal.empty),
            "baseline_sampling_precision_uses_reconstructed_t0_support_with_missingness_indicator": True,
            "zero_origin_sampling_support_sensitivity_completed": bool(not support_sensitivity.empty),
            "occurrence_calibration_summary_completed": bool(not occurrence_calibration.empty),
            "white_noise_residual_diagnostics": True,
            "embedded_log_diffusion_sensitivity": config.log_diffusion_enabled,
            "positive_state_diffusion_sensitivity": config.positive_diffusion_enabled,
            "recursive_long_horizon_claimed": False,
            "causal_climate_claimed": False,
        },
        "final_hyperparameters": jsonable(asdict(final_model.hyper)),
        "final_calibration": {"mode": final_calibration.mode, "q_young": final_calibration.conformal_q_young, "q_mature": final_calibration.conformal_q_mature, "calibration_sites": final_calibration.calibration_sites, "stochastic_source_sites": final_calibration.stochastic_source_sites},
        "discrepancy_audit": final_model.discrepancy.mathematical_audit(),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Workbook: {workbook}")
    print(f"Completed in {runtime:.1f}s: {out}")

if __name__ == "__main__":
    main()
