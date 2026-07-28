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
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_NAME = "PACT-Transition 3.0: zero-aware persistence-anchored process-guided transition model"
MODEL_VERSION = "3.0.0"

Q_MIN, Q_MAX = 1e-4, 5.0
F_MIN, F_MAX = 1e-4, 1.5
H_MIN, H_MAX = 1e-4, 1.5

RAW_CLIMATE_SOURCE = [
    "clim_tmean_c",
    "clim_vpd_mean_kpa",
    "clim_prcp_annualized_mm",
    "clim_deficit_annualized_mm",
]
RAW_CLIMATE_MODEL = ["clim_tmean_c", "clim_vpd_mean_kpa", "log_prcp", "log_deficit"]

BASE_FEATURE_ORDER = [
    "u0_log",
    "v0_log",
    "dt_years",
    "log_dt",
    "event_year",
    "decimalLatitude",
    "decimalLongitude",
    "young_area_precision",
    "mature_area_precision",
    "zero_u0",
    "zero_v0",
]

DEFAULT_FEATURE_WEIGHT_MAP = {
    "u0_log": 2.5,
    "v0_log": 1.2,
    "dt_years": 0.7,
    "log_dt": 0.7,
    "event_year": 0.7,
    "decimalLatitude": 0.7,
    "decimalLongitude": 0.7,
    "young_area_precision": 2.2,
    "mature_area_precision": 0.8,
    "zero_u0": 2.5,
    "zero_v0": 1.2,
    "climate_PC1": 0.8,
    "climate_PC2": 0.8,
}

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
    gate: float = 0.25
    hurdle_point_weight: float = 0.0
    use_mechanism: bool = True
    use_teacher: bool = True
    use_climate: bool = True
    use_coordinates: bool = True
    use_precision_features: bool = True
    use_precision_weights: bool = True
    use_hurdle: bool = True

@dataclass
class Configuration:
    outer_folds: int = 3
    calibration_fraction: float = 0.35
    interval: float = 0.90
    conformal_mode: str = "site_quantile"
    conformal_site_quantile: float = 0.90
    cluster_bootstrap: int = 1000
    paired_bootstrap: int = 1000
    mc_draws: int = 512
    process_trim_quantile: float = 0.975
    integration_steps: int = 16
    mechanism_multistart: int = 4
    mechanism_max_nfev: int = 250
    teacher_folds: int = 3
    teacher_young_trees: int = 240
    teacher_mature_trees: int = 240
    teacher_joint_trees: int = 300
    teacher_max_depth_young: int = 9
    teacher_max_depth_mature: int = 6
    teacher_max_depth_joint: int = 9
    distillation_data_weight: float = 0.35
    hurdle_folds: int = 4
    hurdle_trees: int = 360
    hurdle_min_leaf: int = 8
    probability_clip: float = 0.002
    hurdle_point_weight: float = 0.0
    mechanistic_gate: float = 0.25
    kernel_gammas: tuple[float,...] = (0.003,0.03)
    kernel_weights: tuple[float,...] = (0.70,0.30)
    young_kernel_ridge: float = 0.10
    mature_kernel_ridge: float = 1.00
    kernel_jitter: float = 1e-8
    feature_weight_map: dict[str,float] = field(default_factory=lambda:dict(DEFAULT_FEATURE_WEIGHT_MAP))
    tune_gate: bool = True
    inner_folds: int = 3
    gate_grid: tuple[float,...] = (0.0,0.10,0.25,0.50,0.75,1.0)
    hurdle_weight_grid: tuple[float,...] = (0.0,0.25,0.50,0.75,1.0)
    tuning_zero_weight: float = 0.20
    repeated_holdouts: int = 30
    repeated_test_fraction: float = 0.25
    surrogate_splits: int = 20
    surrogate_top_terms: int = 24
    surrogate_max_iter: int = 10000
    dpi: int = 300
    seed: int = 2026
    profile: str = "full"
    @classmethod
    def for_profile(cls,profile:str,**overrides:Any)->"Configuration":
        profile=profile.lower()
        if profile not in {"smoke","standard","full"}: raise ValueError("profile must be smoke, standard, or full")
        c=cls(profile=profile)
        if profile=="smoke":
            c.outer_folds=2;c.cluster_bootstrap=30;c.paired_bootstrap=30;c.mc_draws=48;c.mechanism_multistart=1;c.mechanism_max_nfev=70;c.teacher_young_trees=30;c.teacher_mature_trees=30;c.teacher_joint_trees=40;c.hurdle_trees=50;c.repeated_holdouts=2;c.surrogate_splits=3;c.dpi=120
        elif profile=="standard":
            c.cluster_bootstrap=500;c.paired_bootstrap=500;c.mc_draws=256;c.mechanism_multistart=2;c.mechanism_max_nfev=180;c.teacher_young_trees=120;c.teacher_mature_trees=120;c.teacher_joint_trees=160;c.hurdle_trees=200;c.repeated_holdouts=10;c.surrogate_splits=10;c.dpi=240
        for k,v in overrides.items():
            if v is not None:setattr(c,k,v)
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

def validate_no_group_overlap(*frames: pd.DataFrame) -> bool:
    groups = [set(frame.siteID.astype(str)) for frame in frames]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if groups[left] & groups[right]:
                return False
    return True

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
    data = (
        data.sort_values(["transition_id", "siteID", "plotID"])
        .drop_duplicates("transition_id")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError("No valid transitions remain after quality control")
    if data.siteID.nunique() < 4:
        raise ValueError("At least four independent sites are required")

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
        pc1_reference = components[0, 0] + components[0, 1] + components[0, 3]
        if pc1_reference < 0:
            orientation[0] = -1.0
        if components[1, 2] * orientation[1] < 0:
            orientation[1] = -1.0
        return cls(scaler=scaler, pca=pca, orientation=orientation)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        scores = self.pca.transform(self.scaler.transform(data[RAW_CLIMATE_MODEL]))
        return scores * self.orientation

    def loading_table(self, label: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        oriented = self.pca.components_ * self.orientation[:, None]
        for index in range(2):
            for feature, loading in zip(RAW_CLIMATE_MODEL, oriented[index]):
                rows.append(
                    {
                        "fit": label,
                        "component": f"PC{index + 1}",
                        "feature": feature,
                        "loading": float(loading),
                        "explained_variance_ratio": float(self.pca.explained_variance_ratio_[index]),
                        "cumulative_explained_variance": float(
                            self.pca.explained_variance_ratio_[: index + 1].sum()
                        ),
                    }
                )
        return pd.DataFrame(rows)

@dataclass
class FeatureTransform:
    feature_names: list[str]
    climate: ClimateTransform | None
    scaler: StandardScaler
    weights: np.ndarray

    @staticmethod
    def selected_features(spec: ModelSpec) -> list[str]:
        selected = []
        for feature in BASE_FEATURE_ORDER:
            if feature in {"decimalLatitude", "decimalLongitude"} and not spec.use_coordinates:
                continue
            if feature in {"young_area_precision", "mature_area_precision"} and not spec.use_precision_features:
                continue
            selected.append(feature)
        if spec.use_climate:
            selected.extend(["climate_PC1", "climate_PC2"])
        return selected

    @classmethod
    def fit(cls, data: pd.DataFrame, spec: ModelSpec, config: Configuration, seed: int) -> "FeatureTransform":
        climate = ClimateTransform.fit(data, seed) if spec.use_climate else None
        names = cls.selected_features(spec)
        raw = cls._raw_matrix(data, names, climate)
        scaler = StandardScaler().fit(raw)
        weights = np.array([config.feature_weight_map.get(name, 1.0) for name in names], dtype=float)
        return cls(feature_names=names, climate=climate, scaler=scaler, weights=weights)

    @staticmethod
    def _raw_matrix(
        data: pd.DataFrame,
        names: Sequence[str],
        climate: ClimateTransform | None,
    ) -> np.ndarray:
        columns: list[np.ndarray] = []
        climate_scores = climate.transform(data) if climate is not None else None
        for name in names:
            if name == "climate_PC1":
                assert climate_scores is not None
                columns.append(climate_scores[:, 0])
            elif name == "climate_PC2":
                assert climate_scores is not None
                columns.append(climate_scores[:, 1])
            else:
                columns.append(data[name].to_numpy(float))
        return np.column_stack(columns)

    def transform(self, data: pd.DataFrame, weighted: bool = True) -> np.ndarray:
        raw = self._raw_matrix(data, self.feature_names, self.climate)
        standardized = np.clip(self.scaler.transform(raw), -6.0, 6.0)
        return standardized * self.weights if weighted else standardized

@dataclass
class MechanisticModel:
    climate: ClimateTransform | None
    parameters: np.ndarray
    integration_steps: int
    use_climate: bool
    fit_cost: float
    fit_success: bool
    active_bounds: np.ndarray

    @staticmethod
    def rates(parameters: np.ndarray, climate_scores: np.ndarray) -> tuple[np.ndarray, ...]:
        q = Q_MIN + (Q_MAX - Q_MIN) * expit(parameters[0] + climate_scores @ parameters[1:3])
        maturation = F_MIN + (F_MAX - F_MIN) * expit(
            parameters[3] + climate_scores @ parameters[4:6]
        )
        mortality = H_MIN + (H_MAX - H_MIN) * expit(
            parameters[6] + climate_scores @ parameters[7:9]
        )
        rho, mu, a, b = np.exp(parameters[9:13])
        return q, maturation, mortality, rho, mu, a, b

    @staticmethod
    def integrate_arrays(
        data: pd.DataFrame,
        climate_scores: np.ndarray,
        parameters: np.ndarray,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        u = np.maximum(data.u0_kha.to_numpy(float), 1e-10)
        v = np.maximum(data.v0_kha.to_numpy(float), 1e-10)
        dt = data.dt_years.to_numpy(float)
        step_count = max(1, int(steps))
        h = dt / step_count
        q, maturation, mortality, rho, mu, a, b = MechanisticModel.rates(parameters, climate_scores)

        for _ in range(step_count):
            du = q + rho * v - (mu + maturation) * u - a * u * u
            dv = maturation * u - mortality * v - b * v * v
            u_euler = np.maximum(u + h * du, 1e-10)
            v_euler = np.maximum(v + h * dv, 1e-10)
            du_euler = q + rho * v_euler - (mu + maturation) * u_euler - a * u_euler * u_euler
            dv_euler = maturation * u_euler - mortality * v_euler - b * v_euler * v_euler
            u = np.maximum(u + 0.5 * h * (du + du_euler), 1e-10)
            v = np.maximum(v + 0.5 * h * (dv + dv_euler), 1e-10)
        return np.log1p(u), np.log1p(v)

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        config: Configuration,
        seed: int,
        use_climate: bool = True,
        use_precision_weights: bool = True,
    ) -> "MechanisticModel":
        rng = np.random.default_rng(seed)
        climate = ClimateTransform.fit(data, seed) if use_climate else None
        climate_scores = climate.transform(data) if climate is not None else np.zeros((len(data), 2))

        def bounded_logit(value: float, lower: float, upper: float) -> float:
            probability = np.clip((value - lower) / (upper - lower), 1e-5, 1 - 1e-5)
            return float(logit(probability))

        initial = np.array(
            [
                bounded_logit(0.08, Q_MIN, Q_MAX),
                0.0,
                0.0,
                bounded_logit(0.08, F_MIN, F_MAX),
                0.0,
                0.0,
                bounded_logit(0.06, H_MIN, H_MAX),
                0.0,
                0.0,
                np.log(0.03),
                np.log(0.04),
                np.log(0.02),
                np.log(0.05),
            ]
        )
        young_weight = np.sqrt(np.clip(data.young_area_precision.to_numpy(float),0.5,2.0)) if use_precision_weights else np.ones(len(data))
        mature_weight = np.sqrt(np.clip(data.mature_area_precision.to_numpy(float),0.5,2.0)) if use_precision_weights else np.ones(len(data))
        young_scale = np.std(data.delta_u_log.to_numpy(float)) + 0.10
        mature_scale = np.std(data.delta_v_log.to_numpy(float)) + 0.05

        def residual(parameters: np.ndarray) -> np.ndarray:
            pred_u, pred_v = cls.integrate_arrays(
                data,
                climate_scores,
                parameters,
                max(8, config.integration_steps - 2),
            )
            data_residual = np.concatenate(
                [
                    (pred_u - data.u1_log.to_numpy(float)) * young_weight / young_scale,
                    (pred_v - data.v1_log.to_numpy(float)) * mature_weight / mature_scale,
                ]
            )
            climate_penalty = 0.08 * np.concatenate(
                [parameters[1:3], parameters[4:6], parameters[7:9]]
            )
            if not use_climate:
                climate_penalty = 2.0 * np.concatenate(
                    [parameters[1:3], parameters[4:6], parameters[7:9]]
                )
            positive_parameter_penalty = 0.05 * (parameters[9:13] - initial[9:13])
            return np.concatenate([data_residual, climate_penalty, positive_parameter_penalty])

        lower = np.array([-8, -3, -3, -8, -3, -3, -8, -3, -3, -6, -6, -8, -8], dtype=float)
        upper = np.array([4, 3, 3, 4, 3, 3, 4, 3, 3, 1, 1, 1, 1], dtype=float)

        best = None
        starts = max(1, int(config.mechanism_multistart))
        for start_index in range(starts):
            candidate = initial.copy()
            if start_index > 0:
                candidate += rng.normal(0.0, 0.25, size=candidate.size)
                candidate = np.minimum(np.maximum(candidate, lower + 1e-5), upper - 1e-5)
            result = least_squares(
                residual,
                candidate,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.70,
                max_nfev=config.mechanism_max_nfev,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        if not np.all(np.isfinite(best.x)):
            raise RuntimeError("Mechanistic parameter estimation returned non-finite values")
        active = (np.isclose(best.x, lower, atol=5e-4) | np.isclose(best.x, upper, atol=5e-4)).astype(int)
        return cls(
            climate=climate,
            parameters=best.x,
            integration_steps=config.integration_steps,
            use_climate=use_climate,
            fit_cost=float(best.cost),
            fit_success=bool(best.success),
            active_bounds=active,
        )

    def climate_scores(self, data: pd.DataFrame) -> np.ndarray:
        if self.climate is None:
            return np.zeros((len(data), 2))
        return self.climate.transform(data)

    def predict(self, data: pd.DataFrame, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.integrate_arrays(
            data,
            self.climate_scores(data),
            self.parameters,
            self.integration_steps if steps is None else steps,
        )

    def parameter_table(self, label: str) -> pd.DataFrame:
        p = self.parameters
        names = [
            "q_intercept",
            "q_PC1",
            "q_PC2",
            "F_intercept",
            "F_PC1",
            "F_PC2",
            "H_intercept",
            "H_PC1",
            "H_PC2",
            "log_rho",
            "log_mu",
            "log_a",
            "log_b",
        ]
        rows = []
        for index, (name, estimate) in enumerate(zip(names, p)):
            rows.append(
                {
                    "fit": label,
                    "parameter": name,
                    "estimate_internal": float(estimate),
                    "estimate_natural": float(np.exp(estimate)) if index >= 9 else float(estimate),
                    "active_bound": int(self.active_bounds[index]),
                    "fit_cost": self.fit_cost,
                    "fit_success": self.fit_success,
                    "uses_climate": self.use_climate,
                }
            )
        return pd.DataFrame(rows)

def fit_teacher_experts(
    data: pd.DataFrame,
    indices: np.ndarray,
    transform: FeatureTransform,
    config: Configuration,
    seed: int,
    use_precision_weights: bool = True,
) -> dict[str, Any]:
    x = transform.transform(data.iloc[indices])
    du = data.delta_u_log.to_numpy(float)
    dv = data.delta_v_log.to_numpy(float)

    young = ExtraTreesRegressor(
        n_estimators=config.teacher_young_trees,
        max_depth=config.teacher_max_depth_young,
        min_samples_leaf=6,
        max_features=0.85,
        criterion="squared_error",
        random_state=seed + 1,
        n_jobs=-1,
    )
    young.fit(
        x,
        du[indices],
        sample_weight=np.clip(data.young_area_precision.to_numpy(float)[indices],0.5,3.0) if use_precision_weights else None,
    )

    mature = ExtraTreesRegressor(
        n_estimators=config.teacher_mature_trees,
        max_depth=config.teacher_max_depth_mature,
        min_samples_leaf=10,
        max_features=0.80,
        criterion="absolute_error",
        random_state=seed + 2,
        n_jobs=-1,
    )
    mature.fit(
        x,
        dv[indices],
        sample_weight=np.clip(data.mature_area_precision.to_numpy(float)[indices],0.5,3.0) if use_precision_weights else None,
    )

    scales = np.std(np.column_stack([du[indices], dv[indices]]), axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    joint = ExtraTreesRegressor(
        n_estimators=config.teacher_joint_trees,
        max_depth=config.teacher_max_depth_joint,
        min_samples_leaf=6,
        max_features=0.85,
        criterion="squared_error",
        random_state=seed + 3,
        n_jobs=-1,
    )
    joint.fit(
        x,
        np.column_stack([du[indices], dv[indices]]) / scales,
        sample_weight=np.sqrt(np.clip(data.young_area_precision.to_numpy(float)[indices],0.5,3.0)*np.clip(data.mature_area_precision.to_numpy(float)[indices],0.5,3.0)) if use_precision_weights else None,
    )
    return {"young": young, "mature": mature, "joint": joint, "scales": scales, "transform": transform}

def predict_teacher_experts(
    data: pd.DataFrame,
    indices: np.ndarray,
    fitted: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    x = fitted["transform"].transform(data.iloc[indices])
    base_u = data.u0_log.to_numpy(float)[indices]
    base_v = data.v0_log.to_numpy(float)[indices]
    stage_u = np.maximum(base_u + fitted["young"].predict(x), 0.0)
    stage_v = np.maximum(base_v + fitted["mature"].predict(x), 0.0)
    joint_delta = fitted["joint"].predict(x) * fitted["scales"]
    joint_u = np.maximum(base_u + joint_delta[:, 0], 0.0)
    joint_v = np.maximum(base_v + joint_delta[:, 1], 0.0)
    return np.column_stack([base_u, stage_u, joint_u]), np.column_stack([stage_v, joint_v])

def young_teacher_weights(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    rmse0 = np.sqrt(np.mean((y - predictions[:, 0]) ** 2)) + 1e-12
    mae0 = np.mean(np.abs(y - predictions[:, 0])) + 1e-12

    def objective(weights: np.ndarray) -> float:
        error = y - predictions @ weights
        return float(
            0.75 * np.sqrt(np.mean(error * error)) / rmse0
            + 0.25 * np.mean(np.abs(error)) / mae0
            + 0.002 * np.sum(weights * weights)
        )

    result = minimize(
        objective,
        np.full(3, 1 / 3),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        scores = [np.sqrt(np.mean((y - predictions[:, column]) ** 2)) for column in range(3)]
        fallback = np.zeros(3)
        fallback[int(np.argmin(scores))] = 1.0
        return fallback
    weights = np.clip(result.x, 0.0, 1.0)
    weights[weights < 0.01] = 0.0
    if weights.sum() <= 0:
        weights[:] = 1 / 3
    return weights / weights.sum()

def mature_teacher_weight(y: np.ndarray, predictions: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 101)
    scores = [
        np.sqrt(np.mean((y - ((1 - weight) * predictions[:, 0] + weight * predictions[:, 1])) ** 2))
        for weight in grid
    ]
    return float(grid[int(np.argmin(scores))])

def cross_fitted_teacher_targets(
    data: pd.DataFrame,
    spec: ModelSpec,
    config: Configuration,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.arange(len(data))
    groups = data.siteID.astype(str).to_numpy()
    splits = min(config.teacher_folds, np.unique(groups).size)
    if splits < 2:
        return data.u1_log.to_numpy(float),data.v1_log.to_numpy(float),{"teacher_used":False,"reason":"fewer_than_two_sites"}

    young_predictions = np.zeros((len(data), 3), dtype=float)
    mature_predictions = np.zeros((len(data), 2), dtype=float)
    for fold, (train, valid) in enumerate(GroupKFold(splits).split(indices, groups=groups), start=1):
        transform = FeatureTransform.fit(data.iloc[train], spec, config, seed + fold * 100)
        fitted = fit_teacher_experts(data,train,transform,config,seed+fold*10000,spec.use_precision_weights)
        young_predictions[valid], mature_predictions[valid] = predict_teacher_experts(data, valid, fitted)

    young_weights = young_teacher_weights(data.u1_log.to_numpy(float), young_predictions)
    mature_joint_weight = mature_teacher_weight(data.v1_log.to_numpy(float), mature_predictions)
    target_u = young_predictions @ young_weights
    target_v = (1 - mature_joint_weight) * mature_predictions[:, 0] + mature_joint_weight * mature_predictions[:, 1]
    audit = {
        "young_persistence_weight": float(young_weights[0]),
        "young_stage_weight": float(young_weights[1]),
        "young_joint_weight": float(young_weights[2]),
        "mature_joint_weight": float(mature_joint_weight),
        "teacher_folds": int(splits),
    }
    return target_u, target_v, audit

@dataclass
class MultiScaleKernelClosure:
    transform: FeatureTransform
    train_features: np.ndarray
    coefficients_base_young: np.ndarray
    coefficients_mechanism_young: np.ndarray
    coefficients_base_mature: np.ndarray
    coefficients_mechanism_mature: np.ndarray
    gammas: tuple[float,...]
    kernel_weights: tuple[float,...]
    ridges: tuple[float,float]
    @staticmethod
    def kernel(left:np.ndarray,right:np.ndarray,gammas:tuple[float,...],weights:tuple[float,...])->np.ndarray:
        if len(gammas)!=len(weights):raise ValueError("kernel_gammas and kernel_weights must have equal length")
        d=cdist(left,right,metric="sqeuclidean")
        out=np.zeros_like(d)
        for gamma,weight in zip(gammas,weights):out+=weight*np.exp(-gamma*d)
        return out
    @classmethod
    def fit(cls,data:pd.DataFrame,target_base_young:np.ndarray,target_base_mature:np.ndarray,mechanism_delta_young:np.ndarray,mechanism_delta_mature:np.ndarray,spec:ModelSpec,config:Configuration,seed:int)->"MultiScaleKernelClosure":
        transform=FeatureTransform.fit(data,spec,config,seed);x=transform.transform(data);k=cls.kernel(x,x,config.kernel_gammas,config.kernel_weights);eye=np.eye(len(data))
        def solve_pair(target:np.ndarray,mechanism:np.ndarray,ridge:float)->tuple[np.ndarray,np.ndarray]:
            matrix=k+(ridge+config.kernel_jitter)*eye
            rhs=np.column_stack([target,mechanism])
            try:solution=cho_solve(cho_factor(matrix,lower=True,check_finite=False),rhs,check_finite=False)
            except (LinAlgError,ValueError):solution=np.linalg.solve(matrix+1e-6*eye,rhs)
            return solution[:,0],solution[:,1]
        bu,mu=solve_pair(target_base_young,mechanism_delta_young,config.young_kernel_ridge);bv,mv=solve_pair(target_base_mature,mechanism_delta_mature,config.mature_kernel_ridge)
        return cls(transform,x,bu,mu,bv,mv,config.kernel_gammas,config.kernel_weights,(config.young_kernel_ridge,config.mature_kernel_ridge))
    def predict(self,data:pd.DataFrame,gate:float)->tuple[np.ndarray,np.ndarray]:
        x=self.transform.transform(data);k=self.kernel(x,self.train_features,self.gammas,self.kernel_weights)
        return k@(self.coefficients_base_young-gate*self.coefficients_mechanism_young),k@(self.coefficients_base_mature-gate*self.coefficients_mechanism_mature)
    def predict_basis(self,data:pd.DataFrame)->dict[str,np.ndarray]:
        x=self.transform.transform(data);k=self.kernel(x,self.train_features,self.gammas,self.kernel_weights)
        return {"base_u":k@self.coefficients_base_young,"mechanism_u":k@self.coefficients_mechanism_young,"base_v":k@self.coefficients_base_mature,"mechanism_v":k@self.coefficients_mechanism_mature}
    def ood_diagnostics(self,data:pd.DataFrame)->pd.DataFrame:
        x=self.transform.transform(data);d=cdist(x,self.train_features,metric="sqeuclidean");s=self.kernel(x,self.train_features,self.gammas,self.kernel_weights);p=s/np.maximum(s.sum(axis=1,keepdims=True),1e-12)
        return pd.DataFrame({"nearest_feature_distance":np.sqrt(d.min(axis=1)),"max_kernel_similarity":s.max(axis=1),"effective_kernel_neighbors":1/np.maximum(np.sum(p*p,axis=1),1e-12)})
    def mathematical_audit(self)->dict[str,float]:
        norms={"base_young":float(np.linalg.norm(self.coefficients_base_young,1)),"mechanism_young":float(np.linalg.norm(self.coefficients_mechanism_young,1)),"base_mature":float(np.linalg.norm(self.coefficients_base_mature,1)),"mechanism_mature":float(np.linalg.norm(self.coefficients_mechanism_mature,1))}
        factor=sum(w*math.sqrt(2*g/math.e) for g,w in zip(self.gammas,self.kernel_weights))*float(np.linalg.norm(self.transform.weights))
        return {**{f"kernel_coefficient_L1_{k}":v for k,v in norms.items()},**{f"finite_Lipschitz_upper_{k}":v*factor for k,v in norms.items()}}

@dataclass
class BinaryProbabilityModel:
    classifiers: list[ExtraTreesClassifier]
    calibrator: Any
    calibration_method: str
    constant_probability: float|None
    raw_oof_brier: float
    calibrated_oof_brier: float
    @staticmethod
    def raw_probability(model:ExtraTreesClassifier,x:np.ndarray)->np.ndarray:
        classes=model.classes_;p=model.predict_proba(x)
        return p[:,int(np.flatnonzero(classes==1)[0])] if 1 in classes else np.zeros(len(x))
    @classmethod
    def fit(cls,x:np.ndarray,y:np.ndarray,groups:np.ndarray,config:Configuration,seed:int)->"BinaryProbabilityModel":
        unique=np.unique(y)
        if unique.size<2:return cls([],None,"constant",float(unique[0]),0.0,0.0)
        n_splits=min(config.hurdle_folds,np.unique(groups).size);oof=np.zeros(len(y));models=[]
        if n_splits<2:
            m=ExtraTreesClassifier(n_estimators=config.hurdle_trees,max_depth=9,min_samples_leaf=config.hurdle_min_leaf,max_features=0.85,class_weight=None,random_state=seed,n_jobs=-1);m.fit(x,y);raw=cls.raw_probability(m,x);return cls([m],None,"identity",None,float(brier_score_loss(y,raw)),float(brier_score_loss(y,raw)))
        for fold,(tr,va) in enumerate(GroupKFold(n_splits).split(x,y,groups),1):
            m=ExtraTreesClassifier(n_estimators=config.hurdle_trees,max_depth=9,min_samples_leaf=config.hurdle_min_leaf,max_features=0.85,class_weight=None,random_state=seed+fold,n_jobs=-1)
            m.fit(x[tr],y[tr]);oof[va]=cls.raw_probability(m,x[va]);models.append(m)
        raw=np.clip(oof,config.probability_clip,1-config.probability_clip)
        if min(np.sum(y==0),np.sum(y==1))>=25 and np.unique(raw).size>=12:
            cal=IsotonicRegression(out_of_bounds="clip").fit(raw,y);method="isotonic";adjusted=cal.predict(raw)
        else:
            cal=LogisticRegression(C=1.0,max_iter=2000).fit(logit(raw).reshape(-1,1),y);method="platt";adjusted=cal.predict_proba(logit(raw).reshape(-1,1))[:,1]
        return cls(models,cal,method,None,float(brier_score_loss(y,raw)),float(brier_score_loss(y,np.clip(adjusted,0,1))))
    def predict_probability(self,x:np.ndarray,clip:float)->np.ndarray:
        if self.constant_probability is not None:return np.full(len(x),self.constant_probability)
        raw=np.mean([self.raw_probability(m,x) for m in self.classifiers],axis=0);raw=np.clip(raw,clip,1-clip)
        if self.calibration_method=="isotonic":p=self.calibrator.predict(raw)
        elif self.calibration_method=="platt":p=self.calibrator.predict_proba(logit(raw).reshape(-1,1))[:,1]
        else:p=raw
        return np.clip(p,clip,1-clip)

@dataclass
class HurdleLayer:
    transform: FeatureTransform
    young: BinaryProbabilityModel
    mature: BinaryProbabilityModel
    probability_clip: float
    @classmethod
    def fit(cls,data:pd.DataFrame,transform:FeatureTransform,config:Configuration,seed:int)->"HurdleLayer":
        x=transform.transform(data);groups=data.siteID.astype(str).to_numpy()
        return cls(transform,BinaryProbabilityModel.fit(x,data.positive_u1.to_numpy(int),groups,config,seed+1),BinaryProbabilityModel.fit(x,data.positive_v1.to_numpy(int),groups,config,seed+2),config.probability_clip)
    def predict(self,data:pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
        x=self.transform.transform(data);return self.young.predict_probability(x,self.probability_clip),self.mature.predict_probability(x,self.probability_clip)
    def audit(self)->dict[str,Any]:
        return {"young_calibration_method":self.young.calibration_method,"mature_calibration_method":self.mature.calibration_method,"young_raw_oof_brier":self.young.raw_oof_brier,"young_calibrated_oof_brier":self.young.calibrated_oof_brier,"mature_raw_oof_brier":self.mature.raw_oof_brier,"mature_calibrated_oof_brier":self.mature.calibrated_oof_brier}

@dataclass
class HybridTransitionModel:
    spec: ModelSpec
    mechanism: MechanisticModel|None
    closure: MultiScaleKernelClosure
    hurdle: HurdleLayer|None
    teacher_audit: dict[str,Any]
    @classmethod
    def fit(cls,data:pd.DataFrame,spec:ModelSpec,config:Configuration,seed:int)->"HybridTransitionModel":
        mechanism=MechanisticModel.fit(data,config,seed+101,spec.use_climate,spec.use_precision_weights) if spec.use_mechanism else None
        if spec.use_teacher:teacher_u,teacher_v,teacher_audit=cross_fitted_teacher_targets(data,spec,config,seed+202)
        else:teacher_u=data.u1_log.to_numpy(float);teacher_v=data.v1_log.to_numpy(float);teacher_audit={"teacher_used":False}
        base_u=data.u0_log.to_numpy(float);base_v=data.v0_log.to_numpy(float);actual_u=data.u1_log.to_numpy(float);actual_v=data.v1_log.to_numpy(float)
        mechanism_u,mechanism_v=mechanism.predict(data) if mechanism is not None else (base_u.copy(),base_v.copy())
        w=config.distillation_data_weight if spec.use_teacher else 1.0
        target_u=(1-w)*teacher_u+w*actual_u-base_u;target_v=(1-w)*teacher_v+w*actual_v-base_v
        closure=MultiScaleKernelClosure.fit(data,target_u,target_v,mechanism_u-base_u,mechanism_v-base_v,spec,config,seed+303)
        hurdle=HurdleLayer.fit(data,closure.transform,config,seed+404) if spec.use_hurdle else None
        if hurdle is not None:teacher_audit={**teacher_audit,**hurdle.audit()}
        return cls(spec,mechanism,closure,hurdle,teacher_audit)
    def components(self,data:pd.DataFrame,gate:float|None=None,hurdle_point_weight:float|None=None)->dict[str,np.ndarray]:
        g=self.spec.gate if gate is None else float(gate);hw=self.spec.hurdle_point_weight if hurdle_point_weight is None else float(hurdle_point_weight)
        base_u=data.u0_log.to_numpy(float);base_v=data.v0_log.to_numpy(float);mechanism_u,mechanism_v=self.mechanism.predict(data) if self.mechanism is not None else (base_u.copy(),base_v.copy())
        closure_u,closure_v=self.closure.predict(data,g);gmu=g*(mechanism_u-base_u);gmv=g*(mechanism_v-base_v);mag_u=np.maximum(base_u+gmu+closure_u,0);mag_v=np.maximum(base_v+gmv+closure_v,0)
        pu,pv=self.hurdle.predict(data) if self.hurdle is not None else (np.ones(len(data)),np.ones(len(data)))
        hu=np.log1p(pu*np.expm1(mag_u));hv=np.log1p(pv*np.expm1(mag_v));pred_u=(1-hw)*mag_u+hw*hu;pred_v=(1-hw)*mag_v+hw*hv
        return {"base_u":base_u,"base_v":base_v,"mechanism_u":mechanism_u,"mechanism_v":mechanism_v,"mechanism_delta_u":mechanism_u-base_u,"mechanism_delta_v":mechanism_v-base_v,"gated_mechanism_u":gmu,"gated_mechanism_v":gmv,"closure_u":closure_u,"closure_v":closure_v,"magnitude_u":mag_u,"magnitude_v":mag_v,"positive_probability_u":pu,"positive_probability_v":pv,"hurdle_expected_u":hu,"hurdle_expected_v":hv,"pred_u":pred_u,"pred_v":pred_v}
    def point_prediction_frame(self,data:pd.DataFrame)->pd.DataFrame:
        c=self.components(data);o=self.closure.ood_diagnostics(data)
        frame=pd.DataFrame({"pred_u_log":c["pred_u"],"pred_v_log":c["pred_v"],"mechanism_u_log":c["mechanism_u"],"mechanism_v_log":c["mechanism_v"],"mechanism_delta_u_log":c["mechanism_delta_u"],"mechanism_delta_v_log":c["mechanism_delta_v"],"gated_mechanism_u_log":c["gated_mechanism_u"],"gated_mechanism_v_log":c["gated_mechanism_v"],"closure_u_log":c["closure_u"],"closure_v_log":c["closure_v"],"magnitude_u_log":c["magnitude_u"],"magnitude_v_log":c["magnitude_v"],"positive_probability_u":c["positive_probability_u"],"positive_probability_v":c["positive_probability_v"],"hurdle_expected_u_log":c["hurdle_expected_u"],"hurdle_expected_v_log":c["hurdle_expected_v"]})
        return pd.concat([frame,o.reset_index(drop=True)],axis=1)

@dataclass
class DirectBenchmark:
    name: str
    transform: FeatureTransform
    young_model: Any
    mature_model: Any

    @classmethod
    def fit(
        cls,
        name: str,
        data: pd.DataFrame,
        spec: ModelSpec,
        config: Configuration,
        seed: int,
    ) -> "DirectBenchmark":
        transform = FeatureTransform.fit(data, spec, config, seed)
        x = transform.transform(data)
        y_u = data.u1_log.to_numpy(float)
        y_v = data.v1_log.to_numpy(float)

        if name == "Extra Trees direct":
            young_model = ExtraTreesRegressor(
                n_estimators=max(100, config.teacher_young_trees),
                max_depth=10,
                min_samples_leaf=6,
                max_features=0.85,
                random_state=seed + 1,
                n_jobs=-1,
            )
            mature_model = ExtraTreesRegressor(
                n_estimators=max(100, config.teacher_mature_trees),
                max_depth=8,
                min_samples_leaf=8,
                max_features=0.85,
                random_state=seed + 2,
                n_jobs=-1,
            )
        elif name == "Histogram boosting direct":
            young_model = HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=seed + 1,
            )
            mature_model = clone(young_model).set_params(random_state=seed + 2)
        elif name == "Ridge direct":
            young_model = Ridge(alpha=10.0)
            mature_model = Ridge(alpha=10.0)
        else:
            raise ValueError(f"Unknown benchmark: {name}")

        young_model.fit(x, y_u)
        mature_model.fit(x, y_v)
        return cls(name=name, transform=transform, young_model=young_model, mature_model=mature_model)

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self.transform.transform(data)
        return (
            np.maximum(self.young_model.predict(x), 0.0),
            np.maximum(self.mature_model.predict(x), 0.0),
        )

def estimate_stochastic_layer(
    data: pd.DataFrame,
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    config: Configuration,
) -> dict[str, float]:
    residual_u = data.u1_log.to_numpy(float) - pred_u
    residual_v = data.v1_log.to_numpy(float) - pred_v
    dt = data.dt_years.to_numpy(float)
    obs_u = 1.0 / (data.young_count_t1.to_numpy(float) + 0.5)
    obs_v = 1.0 / (data.mature_count_t1.to_numpy(float) + 0.5)

    def estimate(residual: np.ndarray, observation_variance: np.ndarray) -> tuple[float, float]:
        squared = residual * residual
        threshold = np.quantile(squared, config.process_trim_quantile)
        keep = np.isfinite(squared) & (squared <= threshold)
        design = np.column_stack([dt[keep], observation_variance[keep]])
        coefficients, _ = nnls(design, squared[keep])
        process_sigma = float(np.clip(math.sqrt(max(coefficients[0], 1e-8)), 0.005, 1.5))
        observation_scale = float(np.clip(math.sqrt(max(coefficients[1], 1e-8)), 0.01, 5.0))
        return process_sigma, observation_scale

    sigma_u, observation_u = estimate(residual_u, obs_u)
    sigma_v, observation_v = estimate(residual_v, obs_v)
    standardized_u = residual_u / np.sqrt(
        np.maximum(sigma_u * sigma_u * dt + observation_u * observation_u * obs_u, 1e-8)
    )
    standardized_v = residual_v / np.sqrt(
        np.maximum(sigma_v * sigma_v * dt + observation_v * observation_v * obs_v, 1e-8)
    )
    correlation = np.corrcoef(np.clip(standardized_u, -4, 4), np.clip(standardized_v, -4, 4))[0, 1]
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "sigma_young": sigma_u,
        "sigma_mature": sigma_v,
        "observation_scale_young": observation_u,
        "observation_scale_mature": observation_v,
        "endpoint_noise_correlation": float(np.clip(correlation, -0.90, 0.90)),
    }

def simulate_endpoint_draws(
    data: pd.DataFrame,
    components: dict[str, np.ndarray],
    stochastic: dict[str, float],
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(data)
    dt = data.dt_years.to_numpy(float)
    root_dt = np.sqrt(dt)

    z1 = rng.normal(size=(draws, n))
    z2_independent = rng.normal(size=(draws, n))
    correlation = stochastic["endpoint_noise_correlation"]
    z2 = correlation * z1 + math.sqrt(max(1 - correlation * correlation, 1e-8)) * z2_independent

    point_u_kha = np.expm1(components["pred_u"])
    point_v_kha = np.expm1(components["pred_v"])
    expected_count_u = np.maximum(point_u_kha * data.area_young_m2.to_numpy(float) / 10.0, 0.0)
    expected_count_v = np.maximum(point_v_kha * data.area_mature_m2.to_numpy(float) / 10.0, 0.0)
    observation_sd_u = stochastic["observation_scale_young"] / np.sqrt(expected_count_u + 0.5)
    observation_sd_v = stochastic["observation_scale_mature"] / np.sqrt(expected_count_v + 0.5)

    observation_u = rng.normal(size=(draws, n)) * observation_sd_u
    observation_v = rng.normal(size=(draws, n)) * observation_sd_v
    magnitude_u = np.maximum(
        components["magnitude_u"]
        + stochastic["sigma_young"] * root_dt * z1
        + observation_u,
        0.0,
    )
    magnitude_v = np.maximum(
        components["magnitude_v"]
        + stochastic["sigma_mature"] * root_dt * z2
        + observation_v,
        0.0,
    )

    positive_u = rng.random(size=(draws, n)) < components["positive_probability_u"]
    positive_v = rng.random(size=(draws, n)) < components["positive_probability_v"]
    path_u = np.where(positive_u, magnitude_u, 0.0)
    path_v = np.where(positive_v, magnitude_v, 0.0)
    audit = {
        "minimum_simulated_log_state": float(min(path_u.min(), path_v.min())),
        "maximum_simulated_log_state": float(max(path_u.max(), path_v.max())),
        "all_draws_finite": bool(np.isfinite(path_u).all() and np.isfinite(path_v).all()),
    }
    return path_u, path_v, audit

def raw_interval_from_draws(
    path_u: np.ndarray,
    path_v: np.ndarray,
    interval: float,
) -> dict[str, np.ndarray]:
    tail = (1.0 - interval) / 2.0
    return {
        "lower_u": np.quantile(path_u, tail, axis=0),
        "upper_u": np.quantile(path_u, 1 - tail, axis=0),
        "lower_v": np.quantile(path_v, tail, axis=0),
        "upper_v": np.quantile(path_v, 1 - tail, axis=0),
    }

DISTRIBUTION_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

def endpoint_distribution_summary(
    data: pd.DataFrame,
    path_u: np.ndarray,
    path_v: np.ndarray,
) -> pd.DataFrame:
    result: dict[str, np.ndarray] = {}
    for key, path, base in [
        ("u", path_u, data.u0_log.to_numpy(float)),
        ("v", path_v, data.v0_log.to_numpy(float)),
    ]:
        original = np.expm1(path)
        result[f"distribution_mean_{key}_log"] = np.mean(path, axis=0)
        result[f"distribution_sd_{key}_log"] = np.std(path, axis=0, ddof=1)
        result[f"distribution_mean_{key}_kha"] = np.mean(original, axis=0)
        result[f"distribution_sd_{key}_kha"] = np.std(original, axis=0, ddof=1)
        for probability in DISTRIBUTION_QUANTILES:
            label = f"q{int(round(probability * 100)):02d}"
            q_log = np.quantile(path, probability, axis=0)
            result[f"distribution_{label}_{key}_log"] = q_log
            result[f"distribution_{label}_{key}_kha"] = np.expm1(q_log)
        p_zero = np.mean(path <= 1e-12, axis=0)
        p_positive = 1.0 - p_zero
        result[f"predictive_probability_zero_{key}"] = p_zero
        result[f"predictive_probability_positive_{key}"] = p_positive
        result[f"predictive_probability_increase_{key}"] = np.mean(path > base[None, :], axis=0)
        result[f"predictive_probability_decrease_{key}"] = np.mean(path < base[None, :], axis=0)
        result[f"predictive_probability_near_persistence_{key}"] = np.mean(
            np.abs(path - base[None, :]) <= 0.05,
            axis=0,
        )
        clipped = np.clip(p_positive, 1e-12, 1.0 - 1e-12)
        result[f"positive_state_entropy_{key}"] = -(
            clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
        )
    correlation = np.empty(path_u.shape[1], dtype=float)
    for index in range(path_u.shape[1]):
        value = np.corrcoef(path_u[:, index], path_v[:, index])[0, 1]
        correlation[index] = value if np.isfinite(value) else 0.0
    result["predictive_draw_correlation_uv"] = correlation
    return pd.DataFrame(result)

def nonconformity_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])

def aggregate_site_scores(
    scores: np.ndarray,
    sites: np.ndarray,
    config: Configuration,
) -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.DataFrame({"siteID": sites.astype(str), "score": scores})
    if config.conformal_mode == "transition":
        return scores, frame.assign(aggregate="transition")
    if config.conformal_mode == "site_max":
        grouped = frame.groupby("siteID", as_index=False).score.max()
    elif config.conformal_mode == "site_quantile":
        grouped = (
            frame.groupby("siteID").score.quantile(config.conformal_site_quantile).reset_index()
        )
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
    score_table: pd.DataFrame
    mode: str

def fit_calibration_state(
    model: HybridTransitionModel,
    calibration: pd.DataFrame,
    config: Configuration,
    seed: int,
) -> CalibrationState:
    components = model.components(calibration)
    stochastic = estimate_stochastic_layer(
        calibration,
        components["pred_u"],
        components["pred_v"],
        config,
    )
    path_u, path_v, _ = simulate_endpoint_draws(
        calibration, components, stochastic, config.mc_draws, seed
    )
    interval = raw_interval_from_draws(path_u, path_v, config.interval)
    score_u = nonconformity_scores(
        calibration.u1_log.to_numpy(float), interval["lower_u"], interval["upper_u"]
    )
    score_v = nonconformity_scores(
        calibration.v1_log.to_numpy(float), interval["lower_v"], interval["upper_v"]
    )
    aggregate_u, table_u = aggregate_site_scores(
        score_u, calibration.siteID.to_numpy(), config
    )
    aggregate_v, table_v = aggregate_site_scores(
        score_v, calibration.siteID.to_numpy(), config
    )
    q_u = finite_quantile(aggregate_u, config.interval)
    q_v = finite_quantile(aggregate_v, config.interval)
    table_u = table_u.rename(columns={"score": "score_young"})
    table_v = table_v.rename(columns={"score": "score_mature"})
    score_table = table_u.merge(
        table_v.drop(columns="aggregate", errors="ignore"), on="siteID", how="outer"
    )
    score_table["conformal_mode"] = config.conformal_mode
    return CalibrationState(
        stochastic=stochastic,
        conformal_q_young=q_u,
        conformal_q_mature=q_v,
        calibration_sites=sorted(calibration.siteID.astype(str).unique().tolist()),
        score_table=score_table,
        mode=config.conformal_mode,
    )

def predict_with_calibration(
    model: HybridTransitionModel,
    data: pd.DataFrame,
    calibration: CalibrationState,
    config: Configuration,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    components = model.components(data)
    path_u, path_v, audit = simulate_endpoint_draws(
        data, components, calibration.stochastic, config.mc_draws, seed
    )
    raw = raw_interval_from_draws(path_u, path_v, config.interval)
    lower_u = np.maximum(raw["lower_u"] - calibration.conformal_q_young, 0.0)
    upper_u = raw["upper_u"] + calibration.conformal_q_young
    lower_v = np.maximum(raw["lower_v"] - calibration.conformal_q_mature, 0.0)
    upper_v = raw["upper_v"] + calibration.conformal_q_mature
    point = model.point_prediction_frame(data)
    point = pd.concat(
        [point.reset_index(drop=True), endpoint_distribution_summary(data, path_u, path_v)],
        axis=1,
    )
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
    for key in ["u", "v"]:
        for prefix in ["lower", "upper", "raw_lower", "raw_upper"]:
            point[f"{prefix}_{key}_kha"] = np.expm1(point[f"{prefix}_{key}_log"])
    return point, audit

def split_core_calibration(
    data: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ensure_probability(fraction, "calibration_fraction")
    groups = data.siteID.astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    core, calibration = next(splitter.split(np.arange(len(data)), groups=groups))
    if set(groups[core]) & set(groups[calibration]):
        raise RuntimeError("Core/calibration site overlap detected")
    return core, calibration

def normalized_pair_rmse(data: pd.DataFrame, pred_u: np.ndarray, pred_v: np.ndarray) -> float:
    rmse_u = np.sqrt(mean_squared_error(data.u1_log, pred_u))
    rmse_v = np.sqrt(mean_squared_error(data.v1_log, pred_v))
    scale_u = np.std(data.u1_log.to_numpy(float)) + 1e-8
    scale_v = np.std(data.v1_log.to_numpy(float)) + 1e-8
    return float(0.5 * (rmse_u / scale_u + rmse_v / scale_v))

def tuning_score(data:pd.DataFrame,pred_u:np.ndarray,pred_v:np.ndarray,zero_weight:float)->tuple[float,float,float]:
    overall=normalized_pair_rmse(data,pred_u,pred_v);parts=[]
    for key,pred in [("u",pred_u),("v",pred_v)]:
        mask=data[f"zero_{key}0"].to_numpy(int)==1
        if mask.sum()>=5:
            scale=np.std(data[f"{key}1_log"].to_numpy(float))+1e-8;parts.append(np.sqrt(mean_squared_error(data.loc[mask,f"{key}1_log"],pred[mask]))/scale)
    zero=float(np.mean(parts)) if parts else overall
    return (1-zero_weight)*overall+zero_weight*zero,overall,zero

def tune_gate_nested(data:pd.DataFrame,base_spec:ModelSpec,config:Configuration,seed:int)->tuple[float,float,pd.DataFrame]:
    groups=data.siteID.astype(str).to_numpy();splits=min(config.inner_folds,np.unique(groups).size)
    if splits<2:return base_spec.gate,base_spec.hurdle_point_weight,pd.DataFrame()
    rows=[]
    for inner,(tr,va) in enumerate(GroupKFold(splits).split(np.arange(len(data)),groups=groups),1):
        train=data.iloc[tr].reset_index(drop=True);valid=data.iloc[va].reset_index(drop=True);model=HybridTransitionModel.fit(train,replace(base_spec,gate=0.0,hurdle_point_weight=0.0),config,seed+inner*100000)
        for gate in config.gate_grid:
            for weight in config.hurdle_weight_grid:
                c=model.components(valid,gate,weight);score,overall,zero=tuning_score(valid,c["pred_u"],c["pred_v"],config.tuning_zero_weight)
                rows.append({"gate":float(gate),"hurdle_point_weight":float(weight),"inner_fold":inner,"objective":score,"overall_normalized_RMSE":overall,"zero_origin_normalized_RMSE":zero})
    table=pd.DataFrame(rows);summary=table.groupby(["gate","hurdle_point_weight"],as_index=False).agg(objective=("objective","mean"),objective_sd=("objective","std"),overall_normalized_RMSE=("overall_normalized_RMSE","mean"),zero_origin_normalized_RMSE=("zero_origin_normalized_RMSE","mean"))
    best=summary.sort_values(["objective","gate","hurdle_point_weight"]).iloc[0];gate=float(best.gate);weight=float(best.hurdle_point_weight);table["selected"]=(table.gate==gate)&(table.hurdle_point_weight==weight);summary["inner_fold"]=0;summary["selected"]=(summary.gate==gate)&(summary.hurdle_point_weight==weight)
    return gate,weight,pd.concat([table,summary],ignore_index=True,sort=False)

def primary_spec(config:Configuration)->ModelSpec:
    return ModelSpec("PACT-Transition",config.mechanistic_gate,config.hurdle_point_weight)

def standard_ablation_specs(config:Configuration,profile:str)->list[ModelSpec]:
    core=[ModelSpec("Closure only (refit)",0.0,0.0,False,True,True,True,True,True,True),ModelSpec("Full without teacher",config.mechanistic_gate,config.hurdle_point_weight,True,False),ModelSpec("Full without climate",config.mechanistic_gate,config.hurdle_point_weight,True,True,False),ModelSpec("Full without coordinates",config.mechanistic_gate,config.hurdle_point_weight,True,True,True,False),ModelSpec("Precision weights only",config.mechanistic_gate,config.hurdle_point_weight,True,True,True,True,False,True),ModelSpec("Precision predictors only",config.mechanistic_gate,config.hurdle_point_weight,True,True,True,True,True,False),ModelSpec("Full without precision",config.mechanistic_gate,config.hurdle_point_weight,True,True,True,True,False,False)]
    return core[:2] if profile=="smoke" else core

def prediction_columns_for_model(name:str)->tuple[str,str]:
    key=slug(name);return f"pred_u_log__{key}",f"pred_v_log__{key}"

def run_outer_cv(data:pd.DataFrame,config:Configuration)->dict[str,pd.DataFrame]:
    ensure_probability(config.interval,"interval");groups=data.siteID.astype(str).to_numpy();indices=np.arange(len(data));n_splits=min(config.outer_folds,np.unique(groups).size)
    predictions=[];audits=[];parameters=[];calibrations=[];tunings=[];rates=[];pcas=[]
    for fold,(train_idx,test_idx) in enumerate(GroupKFold(n_splits).split(indices,groups=groups),1):
        outer=data.iloc[train_idx].reset_index(drop=True);test=data.iloc[test_idx].reset_index(drop=True);cr,ca=split_core_calibration(outer,config.calibration_fraction,config.seed+fold*101);core=outer.iloc[cr].reset_index(drop=True);cal=outer.iloc[ca].reset_index(drop=True)
        if not validate_no_group_overlap(core,cal,test):raise RuntimeError(f"site leakage in fold {fold}")
        base=primary_spec(config);gate,hw,tuning=tune_gate_nested(core,base,config,config.seed+fold*1000000) if config.tune_gate else (base.gate,base.hurdle_point_weight,pd.DataFrame());spec=replace(base,gate=gate,hurdle_point_weight=hw)
        if not tuning.empty:tuning.insert(0,"outer_fold",fold);tunings.append(tuning)
        model=HybridTransitionModel.fit(core,spec,config,config.seed+fold*100000);state=fit_calibration_state(model,cal,config,config.seed+fold*100000+50000);pred,draw_audit=predict_with_calibration(model,test,state,config,config.seed+fold*100000+90000)
        keep=["row_id","transition_id","siteID","plotID","dt_years","log_dt","event_year","decimalLatitude","decimalLongitude","u0_kha","v0_kha","u1_kha","v1_kha","u0_log","v0_log","u1_log","v1_log","zero_u0","zero_v0","positive_u1","positive_v1","young_area_precision","mature_area_precision",*RAW_CLIMATE_MODEL]
        out=test[keep].copy();out.insert(1,"fold",fold);pu,pv=prediction_columns_for_model(spec.name);out[pu]=pred.pred_u_log.to_numpy();out[pv]=pred.pred_v_log.to_numpy()
        for col in pred.columns:
            if col not in {"pred_u_log","pred_v_log"}:out[col]=pred[col].to_numpy()
        pu,pv=prediction_columns_for_model("Persistence");out[pu]=out.u0_log;out[pv]=out.v0_log
        pu,pv=prediction_columns_for_model("Mechanistic ODE only");out[pu]=pred.mechanism_u_log;out[pv]=pred.mechanism_v_log
        pu,pv=prediction_columns_for_model("Hurdle expected point");out[pu]=pred.hurdle_expected_u_log;out[pv]=pred.hurdle_expected_v_log
        for j,ab in enumerate(standard_ablation_specs(config,config.profile),1):
            ab=replace(ab,hurdle_point_weight=hw)
            if ab.use_mechanism:ab=replace(ab,gate=gate)
            m=HybridTransitionModel.fit(core,ab,config,config.seed+fold*100000+j*7000);c=m.components(test);au,av=prediction_columns_for_model(ab.name);out[au]=c["pred_u"];out[av]=c["pred_v"]
        benchmark_spec=replace(spec,name="benchmark")
        for j,name in enumerate(["Extra Trees direct","Histogram boosting direct","Ridge direct"],1):
            b=DirectBenchmark.fit(name,core,benchmark_spec,config,config.seed+fold*100000+50000+j*1000);bu,bv=b.predict(test);cu,cv=prediction_columns_for_model(name);out[cu]=bu;out[cv]=bv
        predictions.append(out);sets=[set(x.siteID.astype(str)) for x in (outer,core,cal,test)]
        audits.append({"fold":fold,"outer_train_rows":len(outer),"core_rows":len(core),"calibration_rows":len(cal),"test_rows":len(test),"outer_train_sites":len(sets[0]),"core_sites":len(sets[1]),"calibration_sites":len(sets[2]),"test_sites":len(sets[3]),"train_test_site_overlap":len(sets[0]&sets[3]),"core_calibration_site_overlap":len(sets[1]&sets[2]),"core_test_site_overlap":len(sets[1]&sets[3]),"calibration_test_site_overlap":len(sets[2]&sets[3]),"selected_gate":gate,"selected_hurdle_point_weight":hw,"conformal_mode":config.conformal_mode,"conformal_q_young":state.conformal_q_young,"conformal_q_mature":state.conformal_q_mature,**draw_audit,**model.closure.mathematical_audit(),**{f"stochastic_{k}":v for k,v in state.stochastic.items()},**{f"hurdle_{k}":v for k,v in (model.hurdle.audit() if model.hurdle else {}).items()}})
        if model.mechanism is not None:
            parameters.append(model.mechanism.parameter_table(f"outer_fold_{fold}"));rates.append(climate_rate_response(model.mechanism,f"outer_fold_{fold}"))
            if model.mechanism.climate is not None:pcas.append(model.mechanism.climate.loading_table(f"outer_fold_{fold}_mechanism"))
        if model.closure.transform.climate is not None:pcas.append(model.closure.transform.climate.loading_table(f"outer_fold_{fold}_closure"))
        ct=state.score_table.copy();ct.insert(0,"fold",fold);ct["q_young"]=state.conformal_q_young;ct["q_mature"]=state.conformal_q_mature;calibrations.append(ct)
    oof=pd.concat(predictions,ignore_index=True).sort_values("row_id").reset_index(drop=True)
    if len(oof)!=len(data) or oof.row_id.duplicated().any():raise RuntimeError("outer CV failed")
    return {"OOF_Predictions":oof,"CV_Audit":pd.DataFrame(audits),"Fold_Parameters":pd.concat(parameters,ignore_index=True) if parameters else pd.DataFrame(),"Calibration_Scores":pd.concat(calibrations,ignore_index=True) if calibrations else pd.DataFrame(),"Gate_Tuning":pd.concat(tunings,ignore_index=True) if tunings else pd.DataFrame(),"Fold_Climate_Rates":pd.concat(rates,ignore_index=True) if rates else pd.DataFrame(),"Fold_PCA_Loadings":pd.concat(pcas,ignore_index=True) if pcas else pd.DataFrame()}

def discover_models(oof: pd.DataFrame) -> dict[str, tuple[str, str]]:
    models: dict[str, tuple[str, str]] = {}
    for column in oof.columns:
        if not column.startswith("pred_u_log__"):
            continue
        key = column.replace("pred_u_log__", "")
        v_column = f"pred_v_log__{key}"
        if v_column in oof.columns:
            models[key] = (column, v_column)
    return models

def metric_values(
    y: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    interval: float = 0.90,
) -> dict[str, float]:
    values = {
        "RMSE_log1p": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE_log1p": float(mean_absolute_error(y, prediction)),
        "R2_log1p": float(r2_score(y, prediction)),
        "Spearman": safe_spearman(y, prediction),
        "Bias_log1p": float(np.mean(prediction - y)),
    }
    if lower is not None and upper is not None:
        values.update(
            {
                "Coverage": float(np.mean((y >= lower) & (y <= upper))),
                "Mean_interval_width_log1p": float(np.mean(upper - lower)),
                "Median_interval_width_log1p": float(np.median(upper - lower)),
                "Interval_score": interval_score(y, lower, upper, interval),
            }
        )
    return values

def model_label_from_key(key: str) -> str:
    mapping = {
        "pact_transition": "PACT-Transition",
        "persistence": "Persistence",
        "mechanistic_ode_only": "Mechanistic ODE only",
        "hurdle_expected_point": "Hurdle expected point",
        "closure_only_refit": "Closure only (refit)",
        "full_without_teacher": "Full without teacher",
        "full_without_climate": "Full without climate",
        "full_without_coordinates": "Full without coordinates",
        "precision_weights_only": "Precision weights only",
        "precision_predictors_only": "Precision predictors only",
        "full_without_precision": "Full without precision",
        "extra_trees_direct": "Extra Trees direct",
        "histogram_boosting_direct": "Histogram boosting direct",
        "ridge_direct": "Ridge direct",
    }
    return mapping.get(key, key.replace("_", " ").title())

def cluster_bootstrap_distribution(
    oof: pd.DataFrame,
    stage: str,
    prediction_column: str,
    repetitions: int,
    seed: int,
) -> dict[str, np.ndarray]:
    key = "u" if stage == "young" else "v"
    y = oof[f"{key}1_log"].to_numpy(float)
    prediction = oof[prediction_column].to_numpy(float)
    sites = oof.siteID.astype(str).to_numpy()
    unique_sites = np.unique(sites)
    site_rows = {site: np.flatnonzero(sites == site) for site in unique_sites}
    rng = np.random.default_rng(seed)
    draws = {"RMSE_log1p": [], "MAE_log1p": [], "R2_log1p": [], "Spearman": [], "Bias_log1p": []}
    for _ in range(repetitions):
        sampled_sites = rng.choice(unique_sites, len(unique_sites), replace=True)
        sampled = np.concatenate([site_rows[site] for site in sampled_sites])
        values = metric_values(y[sampled], prediction[sampled])
        for metric in draws:
            draws[metric].append(values[metric])
    return {metric: np.asarray(values, dtype=float) for metric, values in draws.items()}

def model_metrics_table(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    rows = []
    models = discover_models(oof)
    for model_index, (model_key, columns) in enumerate(models.items(), start=1):
        model_name = model_label_from_key(model_key)
        for stage, key, prediction_column in [
            ("young", "u", columns[0]),
            ("mature", "v", columns[1]),
        ]:
            y = oof[f"{key}1_log"].to_numpy(float)
            prediction = oof[prediction_column].to_numpy(float)
            lower = upper = None
            if model_key == "pact_transition":
                lower = oof[f"lower_{key}_log"].to_numpy(float)
                upper = oof[f"upper_{key}_log"].to_numpy(float)
            estimates = metric_values(y, prediction, lower, upper, config.interval)
            distributions = cluster_bootstrap_distribution(
                oof,
                stage,
                prediction_column,
                config.cluster_bootstrap,
                config.seed + model_index * 100 + (1 if stage == "young" else 2),
            )
            for metric, estimate in estimates.items():
                distribution = distributions.get(metric)
                rows.append(
                    {
                        "model": model_name,
                        "model_key": model_key,
                        "stage": stage,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": float(np.nanquantile(distribution, 0.025))
                        if distribution is not None
                        else np.nan,
                        "ci_upper": float(np.nanquantile(distribution, 0.975))
                        if distribution is not None
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)

def paired_model_comparisons(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    models = discover_models(oof)
    primary = models.get("pact_transition")
    if primary is None:
        return pd.DataFrame()
    sites = oof.siteID.astype(str).to_numpy()
    unique_sites = np.unique(sites)
    site_rows = {site: np.flatnonzero(sites == site) for site in unique_sites}
    rng = np.random.default_rng(config.seed + 700_000)
    rows = []
    for model_key, columns in models.items():
        if model_key == "pact_transition":
            continue
        for stage, key, primary_column, comparison_column in [
            ("young", "u", primary[0], columns[0]),
            ("mature", "v", primary[1], columns[1]),
        ]:
            y = oof[f"{key}1_log"].to_numpy(float)
            primary_pred = oof[primary_column].to_numpy(float)
            comparison_pred = oof[comparison_column].to_numpy(float)
            differences = []
            for _ in range(config.paired_bootstrap):
                sampled_sites = rng.choice(unique_sites, len(unique_sites), replace=True)
                sampled = np.concatenate([site_rows[site] for site in sampled_sites])
                rmse_primary = np.sqrt(mean_squared_error(y[sampled], primary_pred[sampled]))
                rmse_comparison = np.sqrt(mean_squared_error(y[sampled], comparison_pred[sampled]))
                differences.append(rmse_comparison - rmse_primary)
            differences_array = np.asarray(differences)
            point_difference = float(
                np.sqrt(mean_squared_error(y, comparison_pred))
                - np.sqrt(mean_squared_error(y, primary_pred))
            )
            rows.append(
                {
                    "stage": stage,
                    "comparison_model": model_label_from_key(model_key),
                    "RMSE_comparison_minus_primary": point_difference,
                    "ci_lower": float(np.quantile(differences_array, 0.025)),
                    "ci_upper": float(np.quantile(differences_array, 0.975)),
                    "probability_primary_better": float(np.mean(differences_array > 0)),
                }
            )
    return pd.DataFrame(rows)

def fold_metrics(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    rows = []
    primary_u, primary_v = prediction_columns_for_model("PACT-Transition")
    persistence_u, persistence_v = prediction_columns_for_model("Persistence")
    for fold, frame in oof.groupby("fold"):
        for stage, key, prediction_column, persistence_column in [
            ("young", "u", primary_u, persistence_u),
            ("mature", "v", primary_v, persistence_v),
        ]:
            y = frame[f"{key}1_log"].to_numpy(float)
            prediction = frame[prediction_column].to_numpy(float)
            persistence = frame[persistence_column].to_numpy(float)
            values = metric_values(
                y,
                prediction,
                frame[f"lower_{key}_log"].to_numpy(float),
                frame[f"upper_{key}_log"].to_numpy(float),
                config.interval,
            )
            persistence_rmse = float(np.sqrt(mean_squared_error(y, persistence)))
            rows.append(
                {
                    "fold": fold,
                    "stage": stage,
                    **values,
                    "Persistence_RMSE_log1p": persistence_rmse,
                    "RMSE_skill": 1 - values["RMSE_log1p"] / persistence_rmse,
                }
            )
    return pd.DataFrame(rows)

def grouped_skill(oof: pd.DataFrame, group: str, config: Configuration) -> pd.DataFrame:
    primary_u, primary_v = prediction_columns_for_model("PACT-Transition")
    persistence_u, persistence_v = prediction_columns_for_model("Persistence")
    rows = []
    for group_value, frame in oof.groupby(group, observed=True):
        if len(frame) < 3:
            continue
        for stage, key, prediction_column, persistence_column in [
            ("young", "u", primary_u, persistence_u),
            ("mature", "v", primary_v, persistence_v),
        ]:
            y = frame[f"{key}1_log"].to_numpy(float)
            prediction = frame[prediction_column].to_numpy(float)
            persistence = frame[persistence_column].to_numpy(float)
            rmse_model = float(np.sqrt(mean_squared_error(y, prediction)))
            rmse_persistence = float(np.sqrt(mean_squared_error(y, persistence)))
            rows.append(
                {
                    group: group_value,
                    "stage": stage,
                    "n": len(frame),
                    "n_sites": frame.siteID.nunique(),
                    "RMSE_model": rmse_model,
                    "RMSE_persistence": rmse_persistence,
                    "absolute_RMSE_gain": rmse_persistence - rmse_model,
                    "RMSE_skill": 1 - rmse_model / rmse_persistence if rmse_persistence > 0 else np.nan,
                    "MAE_model": float(mean_absolute_error(y, prediction)),
                    "R2_model": float(r2_score(y, prediction)),
                    "Coverage": float(
                        np.mean(
                            (y >= frame[f"lower_{key}_log"].to_numpy(float))
                            & (y <= frame[f"upper_{key}_log"].to_numpy(float))
                        )
                    ),
                    "Mean_interval_width": float(
                        np.mean(
                            frame[f"upper_{key}_log"].to_numpy(float)
                            - frame[f"lower_{key}_log"].to_numpy(float)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)

def horizon_skill(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    frame = oof.copy()
    frame["horizon_bin"] = pd.qcut(frame.dt_years, q=5, duplicates="drop").astype(str)
    result = grouped_skill(frame, "horizon_bin", config)
    ranges = (
        frame.groupby("horizon_bin", observed=True)
        .dt_years.agg(["min", "median", "max"])
        .reset_index()
    )
    return result.merge(ranges, on="horizon_bin", how="left")

def ood_analysis(oof: pd.DataFrame, config: Configuration) -> pd.DataFrame:
    frame = oof.copy()
    frame["ood_bin"] = pd.qcut(
        frame.nearest_feature_distance,
        q=5,
        duplicates="drop",
    ).astype(str)
    result = grouped_skill(frame, "ood_bin", config)
    ranges = (
        frame.groupby("ood_bin", observed=True)
        .nearest_feature_distance.agg(["min", "median", "max"])
        .reset_index()
    )
    return result.merge(ranges, on="ood_bin", how="left")

def interval_calibration_diagnostics(oof:pd.DataFrame,config:Configuration)->pd.DataFrame:
    rows=[]
    for stage,key in [("young","u"),("mature","v")]:
        y=oof[f"{key}1_log"].to_numpy(float)
        for label,lo,hi in [("raw_monte_carlo",f"raw_lower_{key}_log",f"raw_upper_{key}_log"),("site_conformal",f"lower_{key}_log",f"upper_{key}_log")]:
            lower=oof[lo].to_numpy(float);upper=oof[hi].to_numpy(float);covered=(y>=lower)&(y<=upper);site=pd.DataFrame({"siteID":oof.siteID.astype(str),"covered":covered,"width":upper-lower}).groupby("siteID").agg(coverage=("covered","mean"),all_covered=("covered","all"),width=("width","mean"))
            rows.append({"stage":stage,"interval":label,"nominal":config.interval,"transition_coverage":float(covered.mean()),"mean_site_coverage":float(site.coverage.mean()),"simultaneous_site_coverage":float(site.all_covered.mean()),"worst_site_coverage":float(site.coverage.min()),"mean_width":float(np.mean(upper-lower)),"median_width":float(np.median(upper-lower)),"interval_score":interval_score(y,lower,upper,config.interval)})
    return pd.DataFrame(rows)

def transition_regime_analysis(oof:pd.DataFrame,config:Configuration)->pd.DataFrame:
    models=discover_models(oof);full=models.get("pact_transition");persistence=models.get("persistence");closure=models.get("closure_only_refit")
    if full is None or persistence is None:return pd.DataFrame()
    rows=[]
    for stage,key,index in [("young","u",0),("mature","v",1)]:
        start=oof[f"zero_{key}0"].to_numpy(int)==0;end=oof[f"positive_{key}1"].to_numpy(int)==1
        labels=np.select([~start&~end,~start&end,start&~end,start&end],["zero_to_zero","zero_to_positive","positive_to_zero","positive_to_positive"],default="unknown")
        for regime in np.unique(labels):
            mask=labels==regime;y=oof.loc[mask,f"{key}1_log"].to_numpy(float);pf=oof.loc[mask,full[index]].to_numpy(float);pp=oof.loc[mask,persistence[index]].to_numpy(float);rf=float(np.sqrt(mean_squared_error(y,pf)));rp=float(np.sqrt(mean_squared_error(y,pp)));row={"stage":stage,"regime":regime,"transitions":int(mask.sum()),"sites":int(oof.loc[mask,"siteID"].nunique()),"PACT_RMSE":rf,"Persistence_RMSE":rp,"skill_vs_persistence":1-rf/rp if rp>1e-12 else np.nan}
            if closure is not None:
                rc=float(np.sqrt(mean_squared_error(y,oof.loc[mask,closure[index]])));row.update({"Closure_RMSE":rc,"incremental_skill_vs_closure":1-rf/rc if rc>1e-12 else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)

def contribution_evidence(oof:pd.DataFrame,metrics:pd.DataFrame,paired:pd.DataFrame)->pd.DataFrame:
    rows=[];comparisons=[("mechanism","Closure only (refit)"),("climate","Full without climate"),("coordinates","Full without coordinates"),("precision","Full without precision"),("teacher","Full without teacher")]
    for component,comparison in comparisons:
        for stage in ("young","mature"):
            f=metrics[(metrics.model=="PACT-Transition")&(metrics.stage==stage)&(metrics.metric=="RMSE_log1p")];c=metrics[(metrics.model==comparison)&(metrics.stage==stage)&(metrics.metric=="RMSE_log1p")]
            if f.empty or c.empty:continue
            rf=float(f.estimate.iloc[0]);rc=float(c.estimate.iloc[0]);p=paired[(paired.stage==stage)&(paired.comparison_model==comparison)]
            rows.append({"component":component,"stage":stage,"full_RMSE":rf,"comparison_RMSE":rc,"RMSE_gain":rc-rf,"relative_gain":1-rf/rc if rc>0 else np.nan,"bootstrap_probability_full_better":float(p.probability_primary_better.iloc[0]) if not p.empty else np.nan})
    for stage,key in [("young","u"),("mature","v")]:rows.append({"component":"magnitude_decomposition","stage":stage,"full_RMSE":np.nan,"comparison_RMSE":np.nan,"RMSE_gain":float(np.mean(np.abs(oof[f"gated_mechanism_{key}_log"]))),"relative_gain":float(np.mean(np.abs(oof[f"closure_{key}_log"]))),"bootstrap_probability_full_better":np.nan})
    return pd.DataFrame(rows)

def research_gap_evidence(metrics:pd.DataFrame,contribution:pd.DataFrame,regimes:pd.DataFrame,calibration:pd.DataFrame,repeated:pd.DataFrame,surrogate:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for stage in ("young","mature"):
        f=metrics[(metrics.model=="PACT-Transition")&(metrics.stage==stage)&(metrics.metric=="RMSE_log1p")];p=metrics[(metrics.model=="Persistence")&(metrics.stage==stage)&(metrics.metric=="RMSE_log1p")]
        if not f.empty and not p.empty:
            skill=1-float(f.estimate.iloc[0])/float(p.estimate.iloc[0]);rows.append({"gap":"unseen-site transition prediction","stage":stage,"evidence":skill,"status":"supported" if skill>0.05 else "weak","claim":f"Site-blocked RMSE skill over persistence was {skill:.1%}."})
        m=contribution[(contribution.component=="mechanism")&(contribution.stage==stage)]
        if not m.empty:
            gain=float(m.relative_gain.iloc[0]);prob=float(m.bootstrap_probability_full_better.iloc[0]);rows.append({"gap":"incremental value of mechanistic structure","stage":stage,"evidence":gain,"status":"supported" if gain>0.01 and prob>=0.95 else "not supported","claim":f"The ODE added {gain:.1%} RMSE skill beyond the independently refitted closure."})
        c=contribution[(contribution.component=="climate")&(contribution.stage==stage)]
        if not c.empty:
            gain=float(c.relative_gain.iloc[0]);prob=float(c.bootstrap_probability_full_better.iloc[0]);rows.append({"gap":"out-of-site predictive value of realized climate","stage":stage,"evidence":gain,"status":"supported" if gain>0.01 and prob>=0.95 else "not supported","claim":f"Climate added {gain:.1%} RMSE skill after site blocking."})
        z=regimes[(regimes.stage==stage)&(regimes.regime=="zero_to_positive")]
        if not z.empty:
            skill=float(z.skill_vs_persistence.iloc[0]);rows.append({"gap":"zero-to-positive establishment transitions","stage":stage,"evidence":skill,"status":"supported" if skill>0.10 else "weak","claim":f"Skill for zero-to-positive transitions was {skill:.1%}."})
        q=calibration[(calibration.stage==stage)&(calibration.interval=="site_conformal")]
        if not q.empty:
            cov=float(q.simultaneous_site_coverage.iloc[0]);rows.append({"gap":"site-aware predictive uncertainty","stage":stage,"evidence":cov,"status":"supported" if abs(cov-float(q.nominal.iloc[0]))<=0.10 else "partial","claim":f"Simultaneous site coverage was {cov:.1%} at nominal {float(q.nominal.iloc[0]):.0%}."})
        s=surrogate[surrogate.stage==stage]
        if not s.empty:
            r=float(s.grouped_OOF_R2.iloc[0]);n=int(s.claimable_terms.iloc[0]);rows.append({"gap":"interpretable model discrepancy","stage":stage,"evidence":r,"status":"supported" if r>=0.70 and n>0 else "partial","claim":f"The sparse closure surrogate achieved grouped OOF R²={r:.3f} with {n} stable terms."})
    if not repeated.empty:
        for stage,g in repeated.groupby("stage"):
            skill=float(g.RMSE_skill.median());rows.append({"gap":"split robustness","stage":stage,"evidence":skill,"status":"supported" if skill>0 else "weak","claim":f"Median skill across repeated site holdouts was {skill:.1%}."})
    return pd.DataFrame(rows)

def paper_claims(gaps:pd.DataFrame)->pd.DataFrame:
    order={"supported":0,"partial":1,"weak":2,"not supported":3};x=gaps.copy();x["order"]=x.status.map(order).fillna(9);x=x.sort_values(["order","gap","stage"]);x["claim_scope"]=np.where(x.status=="supported","main conclusion",np.where(x.status=="partial","qualified conclusion","boundary or null finding"));return x[["claim_scope","gap","stage","status","claim","evidence"]]

def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in data.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(data[column], errors="coerce")
        clean = values.dropna()
        if clean.empty:
            continue
        q = clean.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        rows.append(
            {
                "variable": column,
                "n": int(clean.size),
                "missing": int(values.isna().sum()),
                "unique": int(clean.nunique()),
                "mean": float(clean.mean()),
                "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
                "min": float(clean.min()),
                "q01": float(q.loc[0.01]),
                "q05": float(q.loc[0.05]),
                "q25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "q75": float(q.loc[0.75]),
                "q95": float(q.loc[0.95]),
                "q99": float(q.loc[0.99]),
                "max": float(clean.max()),
                "skewness": float(clean.skew()) if clean.size > 2 else np.nan,
                "excess_kurtosis": float(clean.kurt()) if clean.size > 3 else np.nan,
                "zero_count": int((clean == 0).sum()),
                "zero_rate": float((clean == 0).mean()),
            }
        )
    return pd.DataFrame(rows)

def categorical_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in data.select_dtypes(exclude=[np.number]).columns:
        values = data[column]
        valid = values.dropna().astype(str)
        if valid.empty:
            continue
        counts = valid.value_counts()
        rows.append(
            {
                "variable": column,
                "n": int(valid.size),
                "missing": int(values.isna().sum()),
                "unique": int(valid.nunique()),
                "most_frequent": counts.index[0],
                "most_frequent_count": int(counts.iloc[0]),
                "most_frequent_rate": float(counts.iloc[0] / valid.size),
            }
        )
    return pd.DataFrame(rows)

def site_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("siteID", as_index=False)
        .agg(
            transitions=("transition_id", "size"),
            plots=("plotID", "nunique"),
            dt_min=("dt_years", "min"),
            dt_median=("dt_years", "median"),
            dt_max=("dt_years", "max"),
            u0_mean=("u0_kha", "mean"),
            u1_mean=("u1_kha", "mean"),
            v0_mean=("v0_kha", "mean"),
            v1_mean=("v1_kha", "mean"),
            young_positive_rate=("positive_u1", "mean"),
            mature_positive_rate=("positive_v1", "mean"),
        )
        .sort_values("siteID")
        .reset_index(drop=True)
    )

def zero_transition_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, prefix in [("young", "u"), ("mature", "v")]:
        start = data[f"{prefix}0_kha"].to_numpy(float) > 0
        end = data[f"{prefix}1_kha"].to_numpy(float) > 0
        labels = np.select(
            [~start & ~end, ~start & end, start & ~end, start & end],
            ["zero_to_zero", "zero_to_positive", "positive_to_zero", "positive_to_positive"],
            default="unclassified",
        )
        counts = pd.Series(labels).value_counts()
        for transition_type, count in counts.items():
            rows.append(
                {
                    "stage": stage,
                    "transition_type": transition_type,
                    "count": int(count),
                    "rate": float(count / len(data)),
                }
            )
    return pd.DataFrame(rows)

def hurdle_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        y = oof[f"positive_{key}1"].to_numpy(int)
        probability = np.clip(oof[f"positive_probability_{key}"].to_numpy(float), 1e-6, 1 - 1e-6)
        row = {
            "stage": stage,
            "positive_rate": float(np.mean(y)),
            "Brier": float(brier_score_loss(y, probability)),
            "Log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "Accuracy_at_0.5": float(np.mean((probability >= 0.5) == y)),
            "ROC_AUC": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def hurdle_reliability(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, key in [("young", "u"), ("mature", "v")]:
        frame = pd.DataFrame(
            {
                "observed": oof[f"positive_{key}1"].to_numpy(int),
                "probability": oof[f"positive_probability_{key}"].to_numpy(float),
            }
        )
        frame["bin"] = pd.cut(
            frame.probability,
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
            duplicates="drop",
        )
        grouped = frame.groupby("bin", observed=True).agg(
            n=("observed", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_positive_rate=("observed", "mean"),
        )
        grouped = grouped.reset_index()
        grouped.insert(0, "stage", stage)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)

def climate_rate_response(mechanism: MechanisticModel, label: str) -> pd.DataFrame:
    grid = np.linspace(-2.5, 2.5, 101)
    rows = []
    for varied in [0, 1]:
        climate = np.zeros((len(grid), 2))
        climate[:, varied] = grid
        q, maturation, mortality, rho, mu, a, b = MechanisticModel.rates(
            mechanism.parameters, climate
        )
        for index, value in enumerate(grid):
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)

def summarize_climate_rates(fold_rates: pd.DataFrame) -> pd.DataFrame:
    if fold_rates.empty:
        return pd.DataFrame()
    rate_columns = ["recruitment_q", "maturation_F", "mature_loss_H"]
    rows = []
    for (component, value), frame in fold_rates.groupby(
        ["varied_component", "component_value"]
    ):
        for rate in rate_columns:
            rows.append(
                {
                    "varied_component": component,
                    "component_value": value,
                    "rate": rate,
                    "median": float(frame[rate].median()),
                    "lower": float(frame[rate].quantile(0.025)),
                    "upper": float(frame[rate].quantile(0.975)),
                    "minimum": float(frame[rate].min()),
                    "maximum": float(frame[rate].max()),
                    "n_fits": len(frame),
                }
            )
    return pd.DataFrame(rows)

def parameter_stability(fold_parameters: pd.DataFrame) -> pd.DataFrame:
    if fold_parameters.empty:
        return pd.DataFrame()
    rows = []
    for parameter, frame in fold_parameters.groupby("parameter"):
        values = frame.estimate_natural.to_numpy(float)
        rows.append(
            {
                "parameter": parameter,
                "n_fits": len(frame),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "relative_range": float((np.max(values) - np.min(values)) / (abs(np.median(values)) + 1e-8)),
                "active_bound_fraction": float(frame.active_bound.mean()),
                "sign_consistency_internal": float(
                    max(
                        np.mean(frame.estimate_internal.to_numpy(float) >= 0),
                        np.mean(frame.estimate_internal.to_numpy(float) <= 0),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)

def surrogate_matrix(frame:pd.DataFrame)->tuple[np.ndarray,list[str]]:
    names=["u0_log","v0_log","log_dt","event_year","decimalLatitude","decimalLongitude","young_area_precision","mature_area_precision","zero_u0","zero_v0",*RAW_CLIMATE_MODEL]
    x=frame[names].to_numpy(float);x[:,3]-=np.mean(x[:,3]);return x,names

def fit_sparse_closure_surrogate(oof:pd.DataFrame,config:Configuration)->tuple[pd.DataFrame,pd.DataFrame]:
    x,names=surrogate_matrix(oof);groups=oof.siteID.astype(str).to_numpy();poly=PolynomialFeatures(2,include_bias=False).fit(x);splits=list(GroupKFold(min(3,np.unique(groups).size)).split(x,groups=groups));grid=[(a,l) for a in np.logspace(-4,-1,8) for l in (0.8,1.0)];terms=poly.get_feature_names_out(names);term_rows=[];audit=[]
    for stage,key in [("young","u"),("mature","v")]:
        y=oof[f"closure_{key}_log"].to_numpy(float);scores=[]
        for alpha,l1 in grid:
            fold=[]
            for tr,va in splits:
                s1=StandardScaler().fit(x[tr]);xp=poly.fit_transform(s1.transform(x[tr]));s2=StandardScaler().fit(xp);m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=config.surrogate_max_iter,random_state=config.seed).fit(s2.transform(xp),y[tr]);pv=poly.transform(s1.transform(x[va]));fold.append(mean_squared_error(y[va],m.predict(s2.transform(pv))))
            scores.append((float(np.mean(fold)),alpha,l1))
        _,alpha,l1=min(scores);oof_pred=np.zeros(len(y))
        for tr,va in splits:
            s1=StandardScaler().fit(x[tr]);xp=poly.fit_transform(s1.transform(x[tr]));s2=StandardScaler().fit(xp);m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=config.surrogate_max_iter,random_state=config.seed).fit(s2.transform(xp),y[tr]);oof_pred[va]=m.predict(s2.transform(poly.transform(s1.transform(x[va]))))
        s1=StandardScaler().fit(x);xp=poly.fit_transform(s1.transform(x));s2=StandardScaler().fit(xp);final=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=config.surrogate_max_iter,random_state=config.seed).fit(s2.transform(xp),y);coef=final.coef_;selected=np.zeros(len(coef));positive=np.zeros(len(coef));negative=np.zeros(len(coef));splitter=GroupShuffleSplit(config.surrogate_splits,test_size=0.25,random_state=config.seed+5000)
        for tr,_ in splitter.split(x,groups=groups):
            a=StandardScaler().fit(x[tr]);p=poly.fit_transform(a.transform(x[tr]));b=StandardScaler().fit(p);m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=config.surrogate_max_iter,random_state=config.seed).fit(b.transform(p),y[tr]);c=m.coef_;nz=np.abs(c)>1e-9;selected+=nz;positive+=c>1e-9;negative+=c<-1e-9
        freq=selected/config.surrogate_splits;sign=np.maximum(positive,negative)/np.maximum(selected,1);table=pd.DataFrame({"stage":stage,"term":terms,"coefficient":coef,"absolute_coefficient":np.abs(coef),"selection_frequency":freq,"sign_stability":sign,"claimable":(freq>=0.70)&(sign>=0.80)&(np.abs(coef)>1e-9)}).sort_values(["claimable","absolute_coefficient"],ascending=[False,False]).head(config.surrogate_top_terms);term_rows.append(table)
        audit.append({"stage":stage,"grouped_OOF_R2":float(r2_score(y,oof_pred)),"grouped_OOF_RMSE":float(np.sqrt(mean_squared_error(y,oof_pred))),"alpha":alpha,"l1_ratio":l1,"nonzero_terms":int(np.sum(np.abs(coef)>1e-9)),"claimable_terms":int(np.sum((freq>=0.70)&(sign>=0.80)&(np.abs(coef)>1e-9))),"candidate_terms":len(coef)})
    return pd.concat(term_rows,ignore_index=True),pd.DataFrame(audit)

def run_repeated_holdouts(data:pd.DataFrame,config:Configuration,gate:float,hurdle_weight:float)->pd.DataFrame:
    if config.repeated_holdouts<=0:return pd.DataFrame()
    rows=[];groups=data.siteID.astype(str).to_numpy();splitter=GroupShuffleSplit(config.repeated_holdouts,test_size=config.repeated_test_fraction,random_state=config.seed+4000000)
    for repeat,(tr,te) in enumerate(splitter.split(np.arange(len(data)),groups=groups),1):
        train=data.iloc[tr].reset_index(drop=True);test=data.iloc[te].reset_index(drop=True);cr,ca=split_core_calibration(train,config.calibration_fraction,config.seed+repeat*123);core=train.iloc[cr].reset_index(drop=True);cal=train.iloc[ca].reset_index(drop=True);spec=replace(primary_spec(config),gate=gate,hurdle_point_weight=hurdle_weight);model=HybridTransitionModel.fit(core,spec,config,config.seed+5000000+repeat*100000);state=fit_calibration_state(model,cal,config,config.seed+5000000+repeat*100000+20000);pred,_=predict_with_calibration(model,test,state,config,config.seed+5000000+repeat*100000+40000)
        for stage,key in [("young","u"),("mature","v")]:
            y=test[f"{key}1_log"].to_numpy(float);p=pred[f"pred_{key}_log"].to_numpy(float);base=test[f"{key}0_log"].to_numpy(float);rmse=float(np.sqrt(mean_squared_error(y,p)));pr=float(np.sqrt(mean_squared_error(y,base)));covered=(y>=pred[f"lower_{key}_log"])&(y<=pred[f"upper_{key}_log"]);site=pd.DataFrame({"siteID":test.siteID.astype(str),"covered":covered}).groupby("siteID").covered.agg(["mean","all"])
            rows.append({"repeat":repeat,"stage":stage,"train_sites":train.siteID.nunique(),"test_sites":test.siteID.nunique(),"RMSE_log1p":rmse,"Persistence_RMSE_log1p":pr,"RMSE_skill":1-rmse/pr,"transition_coverage":float(np.mean(covered)),"mean_site_coverage":float(site["mean"].mean()),"simultaneous_site_coverage":float(site["all"].mean()),"mean_interval_width":float(np.mean(pred[f"upper_{key}_log"]-pred[f"lower_{key}_log"]))})
    return pd.DataFrame(rows)

def fit_final_deployment(data:pd.DataFrame,config:Configuration)->tuple[HybridTransitionModel,CalibrationState,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    cr,ca=split_core_calibration(data,config.calibration_fraction,config.seed+777);core=data.iloc[cr].reset_index(drop=True);cal=data.iloc[ca].reset_index(drop=True);spec=primary_spec(config);tuning=pd.DataFrame()
    if config.tune_gate:
        gate,hw,tuning=tune_gate_nested(core,spec,config,config.seed+8000000);spec=replace(spec,gate=gate,hurdle_point_weight=hw)
    model=HybridTransitionModel.fit(core,spec,config,config.seed+8100000);state=fit_calibration_state(model,cal,config,config.seed+8200000);prediction,_=predict_with_calibration(model,data.reset_index(drop=True),state,config,config.seed+8300000);identity=data[["transition_id","siteID","plotID","dt_years","u0_kha","v0_kha","u1_kha","v1_kha"]].reset_index(drop=True);final=pd.concat([identity,prediction],axis=1)
    for key in ("u","v"):
        for prefix in ("pred","lower","upper","mechanism","magnitude"):
            col=f"{prefix}_{key}_log"
            if col in final:final[f"{prefix}_{key}_kha"]=np.expm1(final[col])
    split=pd.DataFrame({"partition":["core","calibration"],"rows":[len(core),len(cal)],"sites":[core.siteID.nunique(),cal.siteID.nunique()],"site_ids":[json.dumps(sorted(core.siteID.astype(str).unique().tolist())),json.dumps(sorted(cal.siteID.astype(str).unique().tolist()))]})
    return model,state,final,split,tuning

def numerical_audit(
    model: HybridTransitionModel,
    data: pd.DataFrame,
    cv_audit: pd.DataFrame,
    config: Configuration,
) -> pd.DataFrame:
    rows = [
        {
            "item": "all_outer_site_overlaps_zero",
            "value": bool(
                (
                    cv_audit[
                        [
                            "train_test_site_overlap",
                            "core_calibration_site_overlap",
                            "core_test_site_overlap",
                            "calibration_test_site_overlap",
                        ]
                    ]
                    == 0
                ).all().all()
            ),
        },
        {
            "item": "all_outer_endpoint_draws_finite",
            "value": bool(cv_audit.all_draws_finite.all()),
        },
        {
            "item": "minimum_outer_simulated_log_state",
            "value": float(cv_audit.minimum_simulated_log_state.min()),
        },
    ]
    if model.mechanism is not None:
        u_steps, v_steps = model.mechanism.predict(data, steps=config.integration_steps)
        u_double, v_double = model.mechanism.predict(data, steps=2 * config.integration_steps)
        rows.extend(
            [
                {
                    "item": "mean_step_sensitivity_young_log",
                    "value": float(np.mean(np.abs(u_steps - u_double))),
                },
                {
                    "item": "max_step_sensitivity_young_log",
                    "value": float(np.max(np.abs(u_steps - u_double))),
                },
                {
                    "item": "mean_step_sensitivity_mature_log",
                    "value": float(np.mean(np.abs(v_steps - v_double))),
                },
                {
                    "item": "max_step_sensitivity_mature_log",
                    "value": float(np.max(np.abs(v_steps - v_double))),
                },
            ]
        )
    return pd.DataFrame(rows)

def make_figures(oof:pd.DataFrame,metrics:pd.DataFrame,regimes:pd.DataFrame,ood:pd.DataFrame,reliability:pd.DataFrame,calibration:pd.DataFrame,contribution:pd.DataFrame,out:Path,config:Configuration)->pd.DataFrame:
    d=out/"figures";d.mkdir(parents=True,exist_ok=True);files=[];pu,pv=prediction_columns_for_model("PACT-Transition")
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,stage,key,col in zip(axes,["Young","Mature"],["u","v"],[pu,pv]):
        y=oof[f"{key}1_log"].to_numpy(float);p=oof[col].to_numpy(float);limit=max(y.max(),p.max())*1.02;im=ax.hexbin(y,p,gridsize=38,mincnt=1,bins="log");ax.plot([0,limit],[0,limit],"--",linewidth=1);ax.set(xlim=(0,limit),ylim=(0,limit),xlabel="Observed log1p density",ylabel="Predicted log1p density",title=stage);fig.colorbar(im,ax=ax,label="log count")
    fig.tight_layout();name="figure_1_oof_prediction.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Site-blocked out-of-fold prediction"))
    rmse=metrics[metrics.metric=="RMSE_log1p"].copy();order=rmse.groupby("model").estimate.mean().sort_values().index;fig,ax=plt.subplots(figsize=(9,max(5,0.34*len(order))));pos=np.arange(len(order));w=0.36
    for off,stage in [(-w/2,"young"),(w/2,"mature")]:cur=rmse[rmse.stage==stage].set_index("model").reindex(order);ax.barh(pos+off,cur.estimate,height=w,label=stage.capitalize())
    ax.set_yticks(pos,order);ax.invert_yaxis();ax.set(xlabel="RMSE on log1p scale",title="Model and refitted ablation comparison");ax.legend(frameon=False);fig.tight_layout();name="figure_2_model_comparison.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Model and independently refitted ablations"))
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,stage,key in zip(axes,["Young","Mature"],["u","v"]):ax.boxplot([oof[f"gated_mechanism_{key}_log"],oof[f"closure_{key}_log"]],tick_labels=["Gated ODE","Learned closure"],showfliers=False);ax.axhline(0,linestyle="--",linewidth=1);ax.set(ylabel="Contribution on log1p scale",title=stage)
    fig.tight_layout();name="figure_3_component_contribution.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Deployed ODE and closure contributions"))
    if not regimes.empty:
        pivot=regimes.pivot(index="regime",columns="stage",values="skill_vs_persistence").fillna(0);fig,ax=plt.subplots(figsize=(9,5));pivot.mul(100).plot.bar(ax=ax);ax.axhline(0,linestyle="--",linewidth=1);ax.set(ylabel="RMSE skill versus persistence (%)",xlabel="Transition regime",title="Where predictive skill is gained");ax.legend(frameon=False);fig.tight_layout();name="figure_4_transition_regimes.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Skill by zero-positive transition regime"))
    if not ood.empty:
        fig,ax=plt.subplots(figsize=(8,5))
        for stage,g in ood.groupby("stage"):q=g.sort_values("median");ax.plot(q["median"],q.RMSE_model,marker="o",label=stage.capitalize())
        ax.set(xlabel="Nearest-training feature distance",ylabel="RMSE on log1p scale",title="Error under feature-space extrapolation");ax.legend(frameon=False);fig.tight_layout();name="figure_5_ood_error.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Prediction error under extrapolation"))
    if not reliability.empty:
        fig,axes=plt.subplots(1,2,figsize=(10,4.6))
        for ax,stage in zip(axes,["young","mature"]):g=reliability[reliability.stage==stage];ax.plot([0,1],[0,1],"--",linewidth=1);ax.plot(g.mean_predicted_probability,g.observed_positive_rate,marker="o");ax.set(xlim=(0,1),ylim=(0,1),xlabel="Predicted positive probability",ylabel="Observed frequency",title=stage.capitalize())
        fig.tight_layout();name="figure_6_hurdle_calibration.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Cross-fitted hurdle probability calibration"))
    if not calibration.empty:
        q=calibration[calibration.interval=="site_conformal"];fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(q));ax.bar(x-0.22,q.transition_coverage,width=0.22,label="Transition");ax.bar(x,q.mean_site_coverage,width=0.22,label="Mean site");ax.bar(x+0.22,q.simultaneous_site_coverage,width=0.22,label="All transitions at site");ax.axhline(config.interval,linestyle="--",linewidth=1,label="Nominal");ax.set_xticks(x,q.stage.str.capitalize());ax.set_ylim(0,1);ax.set(ylabel="Coverage",title="Site-aware interval calibration");ax.legend(frameon=False);fig.tight_layout();name="figure_7_interval_calibration.png";fig.savefig(d/name,dpi=config.dpi);plt.close(fig);files.append((name,"Transition and site-level interval coverage"))
    return pd.DataFrame([{"figure":n,"title":t,"relative_path":str((d/n).relative_to(out)),"bytes":(d/n).stat().st_size} for n,t in files])

def safe_sheet_name(name: str, used: set[str]) -> str:
    candidate = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Sheet"
    base = candidate
    suffix = 1
    while candidate in used:
        token = f"_{suffix}"
        candidate = base[: 31 - len(token)] + token
        suffix += 1
    used.add(candidate)
    return candidate

def save_excel_workbook(out: Path, tables: dict[str, pd.DataFrame]) -> Path:
    workbook_path = out / "PACT_Transition_Results.xlsx"
    used: set[str] = set()
    index_rows = []
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        number_format = workbook.add_format({"num_format": "0.0000"})
        integer_format = workbook.add_format({"num_format": "0"})
        percent_format = workbook.add_format({"num_format": "0.00%"})
        text_format = workbook.add_format({"text_wrap": True, "valign": "top"})
        ordered = sorted(tables.items(), key=lambda item: (item[0] != "Summary", item[0]))
        for name, table in ordered:
            if table is None:
                continue
            frame = table.copy()
            sheet_name = safe_sheet_name(name, used)
            index_rows.append({"table": name, "sheet": sheet_name, "rows": len(frame), "columns": len(frame.columns)})
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(frame.columns):
                worksheet.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)
            for column_index, column in enumerate(frame.columns):
                worksheet.write(0, column_index, column, header_format)
                series = frame[column]
                width = min(max(len(str(column)) + 2, 11), 36)
                if len(frame):
                    width = min(max(width, int(series.astype(str).head(250).str.len().quantile(0.95)) + 2), 36)
                if pd.api.types.is_integer_dtype(series):
                    cell_format = integer_format
                elif pd.api.types.is_float_dtype(series):
                    lower_name = str(column).lower()
                    cell_format = percent_format if any(token in lower_name for token in ["rate", "coverage", "skill", "probability"]) else number_format
                else:
                    cell_format = text_format
                worksheet.set_column(column_index, column_index, width, cell_format)
        index_frame = pd.DataFrame(index_rows)
        index_sheet = safe_sheet_name("Workbook_Index", used)
        index_frame.to_excel(writer, sheet_name=index_sheet, index=False)
        worksheet = writer.sheets[index_sheet]
        worksheet.freeze_panes(1, 0)
        for column_index, column in enumerate(index_frame.columns):
            worksheet.write(0, column_index, column, header_format)
            worksheet.set_column(column_index, column_index, max(12, min(36, len(column) + 4)))
    return workbook_path

def save_tables(out:Path,tables:dict[str,pd.DataFrame])->Path:
    workbook=save_excel_workbook(out,tables)
    if "OOF_Predictions" in tables:tables["OOF_Predictions"].to_csv(out/"OOF_Predictions.csv.gz",index=False,compression="gzip")
    for name in ("Paper_Claims","Research_Gap_Evidence"):
        if name in tables:tables[name].to_csv(out/f"{name}.csv",index=False)
    return workbook

def build_summary(data: pd.DataFrame, metrics: pd.DataFrame, runtime: float) -> pd.DataFrame:
    rows = [
        {"metric": "model", "value": MODEL_NAME},
        {"metric": "model_version", "value": MODEL_VERSION},
        {"metric": "transitions", "value": len(data)},
        {"metric": "plots", "value": data.plotID.nunique()},
        {"metric": "sites", "value": data.siteID.nunique()},
        {"metric": "runtime_seconds", "value": runtime},
    ]
    primary = metrics[metrics.model == "PACT-Transition"]
    for _, row in primary.iterrows():
        rows.append(
            {
                "metric": f"{row.stage}_{row.metric}",
                "value": row.estimate,
            }
        )
    persistence = metrics[(metrics.model == "Persistence") & (metrics.metric == "RMSE_log1p")]
    primary_rmse = primary[primary.metric == "RMSE_log1p"]
    for stage in ["young", "mature"]:
        p = float(primary_rmse[primary_rmse.stage == stage].estimate.iloc[0])
        b = float(persistence[persistence.stage == stage].estimate.iloc[0])
        rows.append({"metric": f"{stage}_RMSE_skill_vs_persistence", "value": 1 - p / b})
    return pd.DataFrame(rows)

def model_card(config: Configuration) -> pd.DataFrame:
    statements = [
        (
            "Scope",
            "Predicts young and mature forest density at the next observed census endpoint for sites excluded from model fitting.",
        ),
        (
            "Deterministic process",
            "A nonnegative two-stage ODE skeleton encodes recruitment, maturation, loss, and density regulation.",
        ),
        (
            "Data-driven discrepancy",
            "A regularized multiscale Gaussian kernel learns transition structure not represented by the ODE.",
        ),
        (
            "Zero process",
            "Cross-fitted Extra Trees classifiers with held-site probability calibration model zero versus positive endpoints; their contribution to the point estimate is selected inside grouped validation.",
        ),
        (
            "Uncertainty",
            "Monte Carlo endpoint draws combine interval-length process variation, count-dependent observation error, and cross-stage correlation.",
        ),
        (
            "Conformal validity target",
            f"Calibration uses {config.conformal_mode} scores from sites excluded from predictor fitting; the same core-fitted predictor is used on calibration and test sites.",
        ),
        (
            "Not claimed",
            "The software is not an Ito or Stratonovich path solver, does not identify causal climate effects, and is not validated for recursive long-horizon simulation.",
        ),
        (
            "Deployment caveat",
            "Climate covariates summarized over the census interval represent conditional hindcast information unless replaced by an operational climate forecast or scenario.",
        ),
    ]
    return pd.DataFrame(statements, columns=["item", "statement"])

def parse_arguments()->argparse.Namespace:
    p=argparse.ArgumentParser(description="PACT-Transition 3.0")
    p.add_argument("--data",default=None);p.add_argument("--out",default="results_pact_transition_v3");p.add_argument("--profile",choices=["smoke","standard","full"],default="full");p.add_argument("--folds",type=int,default=None);p.add_argument("--bootstrap",type=int,default=None);p.add_argument("--paired-bootstrap",type=int,default=None);p.add_argument("--mc-draws",type=int,default=None);p.add_argument("--steps",type=int,default=None);p.add_argument("--seed",type=int,default=2026);p.add_argument("--dpi",type=int,default=None);p.add_argument("--calibration-fraction",type=float,default=None);p.add_argument("--interval",type=float,default=None);p.add_argument("--conformal-mode",choices=["site_max","site_quantile","transition"],default=None);p.add_argument("--repeated-holdouts",type=int,default=None);p.add_argument("--no-tuning",action="store_true")
    args,_=p.parse_known_args();return args

def resolve_data_path(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input data file does not exist: {candidate}")
        return candidate
    directories = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    preferred = ["data(3).csv", "data.csv", "dataset.csv"]
    candidates: list[Path] = []
    for directory in directories:
        candidates.extend(directory / name for name in preferred if (directory / name).exists())
        candidates.extend(sorted(directory.glob("*.csv")))
    seen: set[Path] = set()
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
    raise FileNotFoundError("No compatible CSV was found. Put data(3).csv beside the script or use --data PATH.")

def main()->None:
    args=parse_arguments();start=time.time();data_path=resolve_data_path(args.data);out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    config=Configuration.for_profile(args.profile,outer_folds=args.folds,cluster_bootstrap=args.bootstrap,paired_bootstrap=args.paired_bootstrap,mc_draws=args.mc_draws,integration_steps=args.steps,seed=args.seed,dpi=args.dpi,calibration_fraction=args.calibration_fraction,interval=args.interval,conformal_mode=args.conformal_mode,repeated_holdouts=args.repeated_holdouts)
    if args.no_tuning:config.tune_gate=False
    data=prepare_data(data_path);print(f"{len(data)} transitions | {data.plotID.nunique()} plots | {data.siteID.nunique()} sites | profile={config.profile}")
    evaluation=run_outer_cv(data,config);oof=evaluation["OOF_Predictions"];metrics=model_metrics_table(oof,config);paired=paired_model_comparisons(oof,config);folds=fold_metrics(oof,config);sites=grouped_skill(oof,"siteID",config);horizon=horizon_skill(oof,config);ood=ood_analysis(oof,config);hurdle=hurdle_metrics(oof);reliability=hurdle_reliability(oof);calibration=interval_calibration_diagnostics(oof,config);regimes=transition_regime_analysis(oof,config);contribution=contribution_evidence(oof,metrics,paired);climate=summarize_climate_rates(evaluation["Fold_Climate_Rates"]);stability=parameter_stability(evaluation["Fold_Parameters"]);symbolic_terms,symbolic_audit=fit_sparse_closure_surrogate(oof,config)
    selected_gate=float(evaluation["CV_Audit"].selected_gate.median());selected_weight=float(evaluation["CV_Audit"].selected_hurdle_point_weight.median());repeated=run_repeated_holdouts(data,config,selected_gate,selected_weight);gaps=research_gap_evidence(metrics,contribution,regimes,calibration,repeated,symbolic_audit);claims=paper_claims(gaps)
    final_model,final_calibration,final_prediction,final_split,final_tuning=fit_final_deployment(data,config);numerical=numerical_audit(final_model,data,evaluation["CV_Audit"],config);runtime=time.time()-start;summary=build_summary(data,metrics,runtime);descriptive=descriptive_statistics(data[[c for c in REQUIRED if c in data.columns]+["u0_log","v0_log","u1_log","v1_log","delta_u_log","delta_v_log","log_dt"]]);final_parameters=final_model.mechanism.parameter_table("final_deployment") if final_model.mechanism is not None else pd.DataFrame();final_rates=climate_rate_response(final_model.mechanism,"final_deployment") if final_model.mechanism is not None else pd.DataFrame();run_audit=pd.DataFrame({"item":["model","version","data","sha256","python","numpy","pandas","scipy","scikit_learn","platform","configuration","final_spec","teacher_hurdle_audit","calibration"],"value":[MODEL_NAME,MODEL_VERSION,str(data_path),sha256(data_path),sys.version,np.__version__,pd.__version__,scipy.__version__,sklearn.__version__,platform.platform(),json.dumps(jsonable(asdict(config)),ensure_ascii=False),json.dumps(jsonable(asdict(final_model.spec)),ensure_ascii=False),json.dumps(jsonable(final_model.teacher_audit),ensure_ascii=False),json.dumps({"mode":final_calibration.mode,"q_young":final_calibration.conformal_q_young,"q_mature":final_calibration.conformal_q_mature,"sites":final_calibration.calibration_sites},ensure_ascii=False)]})
    tables={"Summary":summary,"Paper_Claims":claims,"Research_Gap_Evidence":gaps,"Model_Metrics":metrics,"Paired_Model_Comparisons":paired,"Component_Evidence":contribution,"Transition_Regimes":regimes,"Repeated_Holdouts":repeated,"Fold_Metrics":folds,"Site_Skill":sites,"Horizon_Skill":horizon,"OOD_Analysis":ood,"Interval_Calibration":calibration,"Hurdle_Metrics":hurdle,"Hurdle_Reliability":reliability,"Gate_Hurdle_Tuning":evaluation["Gate_Tuning"],"CV_Audit":evaluation["CV_Audit"],"Parameter_Stability":stability,"Climate_Rate_Stability":climate,"Symbolic_Closure_Terms":symbolic_terms,"Symbolic_Closure_Audit":symbolic_audit,"Descriptive_Statistics":descriptive,"Final_Parameters":final_parameters,"Final_Climate_Rates":final_rates,"Final_Split":final_split,"Final_Tuning":final_tuning,"Final_Calibration_Scores":final_calibration.score_table,"OOF_Predictions":oof,"Final_Predictions":final_prediction,"Numerical_Audit":numerical,"Model_Card":model_card(config),"Run_Audit":run_audit}
    figures=make_figures(oof,metrics,regimes,ood,reliability,calibration,contribution,out,config);tables["Figures_Manifest"]=figures;workbook=save_tables(out,tables);joblib.dump({"model_name":MODEL_NAME,"version":MODEL_VERSION,"configuration":asdict(config),"model":final_model,"calibration":final_calibration,"data_sha256":sha256(data_path)},out/"pact_transition_model.joblib",compress=3)
    (out/"paper_results.txt").write_text("\n".join(claims.claim.astype(str)),encoding="utf-8");manifest={"model":MODEL_NAME,"version":MODEL_VERSION,"data":str(data_path),"data_sha256":sha256(data_path),"configuration":jsonable(asdict(config)),"selected_outer_gate_median":selected_gate,"selected_outer_hurdle_weight_median":selected_weight,"runtime_seconds":runtime,"guarantees":{"site_blocked_outer_test":True,"calibration_sites_excluded_from_predictor_fit":True,"same_predictor_for_calibration_and_test":True,"refitted_ablations":True,"cross_fitted_probability_calibration":True,"site_level_interval_reporting":True,"recursive_long_horizon_claimed":False,"causal_climate_claimed":False},"final_calibration":{"mode":final_calibration.mode,"q_young":final_calibration.conformal_q_young,"q_mature":final_calibration.conformal_q_mature},"kernel_audit":final_model.closure.mathematical_audit()};(out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(summary.to_string(index=False));print(f"Workbook: {workbook}");print(f"Completed in {runtime:.1f}s: {out}")

if __name__=="__main__":main()
