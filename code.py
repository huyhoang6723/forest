from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import solve
from scipy.optimize import least_squares, minimize, nnls
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Ill-conditioned matrix")

MODEL_NAME = "PACT-USDE: persistence-anchored universal stochastic differential equation"
RAW_CLIMATE = ["clim_tmean_c", "clim_vpd_mean_kpa", "log_prcp", "log_deficit"]
BASE_FEATURES = [
    "u0_log", "v0_log", "dt_years", "log_dt", "year_centered",
    "decimalLatitude", "decimalLongitude", "young_area_precision",
    "mature_area_precision", "zero_u0", "zero_v0"
]
FEATURE_NAMES = BASE_FEATURES + ["climate_PC1", "climate_PC2"]
FEATURE_WEIGHTS = np.array([2.5, 1.2, 0.7, 0.7, 0.7, 0.7, 0.7, 2.2, 0.8, 2.5, 1.2, 0.8, 0.8])
REQUIRED = [
    "transition_id", "siteID", "plotID", "dt_years", "event_year",
    "decimalLatitude", "decimalLongitude", "u0_kha", "v0_kha", "u1_kha", "v1_kha",
    "young_area_precision", "mature_area_precision", "area_young_m2", "area_mature_m2",
    "young_count_t0", "mature_count_t0", "young_count_t1", "mature_count_t1",
    "clim_tmean_c", "clim_vpd_mean_kpa", "clim_prcp_annualized_mm",
    "clim_deficit_annualized_mm"
]
Q_MIN, Q_MAX = 1e-4, 5.0
F_MIN, F_MAX = 1e-4, 1.5
H_MIN, H_MAX = 1e-4, 1.5


@dataclass
class Configuration:
    folds: int = 3
    calibration_fraction: float = 0.20
    interval: float = 0.90
    bootstrap: int = 1000
    mc_paths: int = 128
    integration_steps: int = 12
    teacher_young_trees: int = 100
    teacher_mature_trees: int = 100
    teacher_joint_trees: int = 140
    distillation_data_weight: float = 0.35
    mechanistic_gate: float = 0.25
    kernel_gammas: tuple[float, ...] = (0.003, 0.03)
    kernel_weights: tuple[float, ...] = (0.70, 0.30)
    young_kernel_ridge: float = 0.10
    mature_kernel_ridge: float = 1.00
    seed: int = 2026
    dpi: int = 300
    conformal_site_shift_margin: float = 0.03


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_quantile(values: np.ndarray, probability: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    rank = min(max(int(math.ceil((x.size + 1) * probability)), 1), x.size)
    return float(np.partition(x, rank - 1)[rank - 1])


def prepare_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in d.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    numeric = [column for column in REQUIRED if column not in {"transition_id", "siteID", "plotID"}]
    for column in numeric:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED).copy()
    d = d[
        (d.dt_years > 0)
        & (d.u0_kha >= 0) & (d.v0_kha >= 0)
        & (d.u1_kha >= 0) & (d.v1_kha >= 0)
        & (d.area_young_m2 > 0) & (d.area_mature_m2 > 0)
    ].copy()
    d = d.sort_values(["transition_id", "siteID", "plotID"]).drop_duplicates("transition_id").reset_index(drop=True)
    if d.empty:
        raise ValueError("No valid transitions remain after quality control")
    d["u0_log"] = np.log1p(d.u0_kha)
    d["v0_log"] = np.log1p(d.v0_kha)
    d["u1_log"] = np.log1p(d.u1_kha)
    d["v1_log"] = np.log1p(d.v1_kha)
    d["delta_u_log"] = d.u1_log - d.u0_log
    d["delta_v_log"] = d.v1_log - d.v0_log
    d["log_dt"] = np.log(d.dt_years)
    d["log_prcp"] = np.log1p(d.clim_prcp_annualized_mm.clip(lower=0))
    d["log_deficit"] = np.log1p(d.clim_deficit_annualized_mm.clip(lower=0))
    d["year_centered"] = d.event_year - d.event_year.mean()
    d["zero_u0"] = (d.u0_kha == 0).astype(float)
    d["zero_v0"] = (d.v0_kha == 0).astype(float)
    return d


@dataclass
class ClimateTransform:
    scaler: StandardScaler
    pca: PCA

    @classmethod
    def fit(cls, d: pd.DataFrame) -> "ClimateTransform":
        scaler = StandardScaler().fit(d[RAW_CLIMATE])
        pca = PCA(n_components=2, random_state=2026).fit(scaler.transform(d[RAW_CLIMATE]))
        return cls(scaler=scaler, pca=pca)

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(d[RAW_CLIMATE]))


@dataclass
class FeatureTransform:
    climate: ClimateTransform
    scaler: StandardScaler

    @classmethod
    def fit(cls, d: pd.DataFrame) -> "FeatureTransform":
        climate = ClimateTransform.fit(d)
        raw = np.column_stack([d[BASE_FEATURES].to_numpy(float), climate.transform(d)])
        scaler = StandardScaler().fit(raw)
        return cls(climate=climate, scaler=scaler)

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        raw = np.column_stack([d[BASE_FEATURES].to_numpy(float), self.climate.transform(d)])
        return np.clip(self.scaler.transform(raw), -6.0, 6.0) * FEATURE_WEIGHTS


@dataclass
class MechanisticModel:
    climate: ClimateTransform
    parameters: np.ndarray
    integration_steps: int

    @staticmethod
    def rates(parameters: np.ndarray, climate_scores: np.ndarray) -> tuple[np.ndarray, ...]:
        q = Q_MIN + (Q_MAX - Q_MIN) * expit(parameters[0] + climate_scores @ parameters[1:3])
        maturation = F_MIN + (F_MAX - F_MIN) * expit(parameters[3] + climate_scores @ parameters[4:6])
        mortality = H_MIN + (H_MAX - H_MIN) * expit(parameters[6] + climate_scores @ parameters[7:9])
        rho, mu, a, b = np.exp(parameters[9:13])
        return q, maturation, mortality, rho, mu, a, b

    @staticmethod
    def integrate_arrays(
        d: pd.DataFrame,
        climate_scores: np.ndarray,
        parameters: np.ndarray,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        u = np.maximum(d.u0_kha.to_numpy(float), 1e-8)
        v = np.maximum(d.v0_kha.to_numpy(float), 1e-8)
        dt = d.dt_years.to_numpy(float)
        h = dt / max(1, int(steps))
        q, maturation, mortality, rho, mu, a, b = MechanisticModel.rates(parameters, climate_scores)
        for _ in range(max(1, int(steps))):
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
    def fit(cls, d: pd.DataFrame, integration_steps: int) -> "MechanisticModel":
        climate = ClimateTransform.fit(d)
        climate_scores = climate.transform(d)
        initial = np.array([
            logit(0.08 / Q_MAX), 0.0, 0.0,
            logit(0.08 / F_MAX), 0.0, 0.0,
            logit(0.06 / H_MAX), 0.0, 0.0,
            np.log(0.03), np.log(0.04), np.log(0.02), np.log(0.05),
        ])
        young_weight = np.sqrt(np.clip(d.young_area_precision.to_numpy(float), 0.5, 2.0))
        mature_weight = np.sqrt(np.clip(d.mature_area_precision.to_numpy(float), 0.5, 2.0))
        young_scale = np.std(d.delta_u_log.to_numpy(float)) + 0.10
        mature_scale = np.std(d.delta_v_log.to_numpy(float)) + 0.05

        def residual(parameters: np.ndarray) -> np.ndarray:
            pred_u, pred_v = cls.integrate_arrays(d, climate_scores, parameters, max(8, integration_steps - 2))
            data_residual = np.concatenate([
                (pred_u - d.u1_log.to_numpy(float)) * young_weight / young_scale,
                (pred_v - d.v1_log.to_numpy(float)) * mature_weight / mature_scale,
            ])
            regularization = np.concatenate([
                0.08 * parameters[1:3], 0.08 * parameters[4:6], 0.08 * parameters[7:9],
                0.05 * (parameters[9:13] - initial[9:13]),
            ])
            return np.concatenate([data_residual, regularization])

        lower = np.array([-8, -3, -3, -8, -3, -3, -8, -3, -3, -6, -6, -8, -8], dtype=float)
        upper = np.array([4, 3, 3, 4, 3, 3, 4, 3, 3, 1, 1, 1, 1], dtype=float)
        result = least_squares(
            residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=0.70,
            max_nfev=160, xtol=1e-7, ftol=1e-7, gtol=1e-7,
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("Mechanistic parameter estimation returned non-finite values")
        return cls(climate=climate, parameters=result.x, integration_steps=integration_steps)

    def predict(self, d: pd.DataFrame, steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.integrate_arrays(
            d, self.climate.transform(d), self.parameters,
            self.integration_steps if steps is None else steps,
        )

    def parameter_table(self, label: str) -> pd.DataFrame:
        p = self.parameters
        rows = [
            ("q_intercept", p[0]), ("q_PC1", p[1]), ("q_PC2", p[2]),
            ("F_intercept", p[3]), ("F_PC1", p[4]), ("F_PC2", p[5]),
            ("H_intercept", p[6]), ("H_PC1", p[7]), ("H_PC2", p[8]),
            ("rho", np.exp(p[9])), ("mu", np.exp(p[10])),
            ("young_density_regulation_a", np.exp(p[11])),
            ("mature_density_regulation_b", np.exp(p[12])),
        ]
        return pd.DataFrame({"fit": label, "parameter": [x[0] for x in rows], "estimate": [x[1] for x in rows]})


@dataclass
class TeacherConfiguration:
    young_trees: int
    mature_trees: int
    joint_trees: int


def fit_teacher_experts(d: pd.DataFrame, indices: np.ndarray, transform: FeatureTransform, config: TeacherConfiguration, seed: int) -> dict[str, Any]:
    x = transform.transform(d.iloc[indices])
    du = d.delta_u_log.to_numpy(float)
    dv = d.delta_v_log.to_numpy(float)
    young = ExtraTreesRegressor(
        n_estimators=config.young_trees, max_depth=9, min_samples_leaf=6,
        max_features=0.85, criterion="squared_error", random_state=seed + 1, n_jobs=-1,
    )
    young.fit(x, du[indices], sample_weight=np.clip(d.young_area_precision.to_numpy(float)[indices], 0.5, 3.0))
    mature = ExtraTreesRegressor(
        n_estimators=config.mature_trees, max_depth=6, min_samples_leaf=10,
        max_features=0.80, criterion="absolute_error", random_state=seed + 2, n_jobs=-1,
    )
    mature.fit(x, dv[indices], sample_weight=np.clip(d.mature_area_precision.to_numpy(float)[indices], 0.5, 3.0))
    scales = np.std(np.column_stack([du[indices], dv[indices]]), axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    joint = ExtraTreesRegressor(
        n_estimators=config.joint_trees, max_depth=9, min_samples_leaf=6,
        max_features=0.85, criterion="squared_error", random_state=seed + 3, n_jobs=-1,
    )
    joint.fit(
        x, np.column_stack([du[indices], dv[indices]]) / scales,
        sample_weight=np.sqrt(
            np.clip(d.young_area_precision.to_numpy(float)[indices], 0.5, 3.0)
            * np.clip(d.mature_area_precision.to_numpy(float)[indices], 0.5, 3.0)
        ),
    )
    return {"young": young, "mature": mature, "joint": joint, "scales": scales, "transform": transform}


def predict_teacher_experts(d: pd.DataFrame, indices: np.ndarray, fitted: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = fitted["transform"].transform(d.iloc[indices])
    base_u = d.u0_log.to_numpy(float)[indices]
    base_v = d.v0_log.to_numpy(float)[indices]
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
        return (
            0.75 * np.sqrt(np.mean(error * error)) / rmse0
            + 0.25 * np.mean(np.abs(error)) / mae0
            + 0.002 * np.sum(weights * weights)
        )

    result = minimize(
        objective, np.full(3, 1 / 3), method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        score = [np.sqrt(np.mean((y - predictions[:, column]) ** 2)) for column in range(3)]
        weights = np.zeros(3)
        weights[int(np.argmin(score))] = 1.0
        return weights
    weights = np.clip(result.x, 0.0, 1.0)
    weights[weights < 0.01] = 0.0
    return weights / weights.sum()


def mature_teacher_weight(y: np.ndarray, predictions: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 101)
    scores = [
        np.sqrt(np.mean((y - ((1 - weight) * predictions[:, 0] + weight * predictions[:, 1])) ** 2))
        for weight in grid
    ]
    return float(grid[int(np.argmin(scores))])


def cross_fitted_teacher_targets(d: pd.DataFrame, config: TeacherConfiguration, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.arange(len(d))
    groups = d.siteID.to_numpy()
    splits = min(3, np.unique(groups).size)
    if splits < 2:
        raise ValueError("At least two sites are required for cross-fitted distillation")
    young_predictions = np.zeros((len(d), 3), dtype=float)
    mature_predictions = np.zeros((len(d), 2), dtype=float)
    for fold, (train, valid) in enumerate(GroupKFold(splits).split(indices, groups=groups), start=1):
        transform = FeatureTransform.fit(d.iloc[train])
        fitted = fit_teacher_experts(d, train, transform, config, seed + fold * 10000)
        young_predictions[valid], mature_predictions[valid] = predict_teacher_experts(d, valid, fitted)
    young_weights = young_teacher_weights(d.u1_log.to_numpy(float), young_predictions)
    mature_joint_weight = mature_teacher_weight(d.v1_log.to_numpy(float), mature_predictions)
    target_u = young_predictions @ young_weights
    target_v = (1 - mature_joint_weight) * mature_predictions[:, 0] + mature_joint_weight * mature_predictions[:, 1]
    audit = {
        "young_persistence_weight": float(young_weights[0]),
        "young_stage_weight": float(young_weights[1]),
        "young_joint_weight": float(young_weights[2]),
        "mature_joint_weight": float(mature_joint_weight),
    }
    return target_u, target_v, audit


@dataclass
class MultiScaleKernelClosure:
    transform: FeatureTransform
    train_features: np.ndarray
    coefficients_young: np.ndarray
    coefficients_mature: np.ndarray
    gammas: tuple[float, ...]
    kernel_weights: tuple[float, ...]
    feature_names: list[str]

    @staticmethod
    def kernel(left: np.ndarray, right: np.ndarray, gammas: tuple[float, ...], weights: tuple[float, ...]) -> np.ndarray:
        distance = cdist(left, right, metric="sqeuclidean")
        output = np.zeros_like(distance)
        for gamma, weight in zip(gammas, weights):
            output += weight * np.exp(-gamma * distance)
        return output

    @classmethod
    def fit(
        cls,
        d: pd.DataFrame,
        target_young: np.ndarray,
        target_mature: np.ndarray,
        config: Configuration,
    ) -> "MultiScaleKernelClosure":
        transform = FeatureTransform.fit(d)
        x = transform.transform(d)
        kernel = cls.kernel(x, x, config.kernel_gammas, config.kernel_weights)
        young_matrix = kernel + config.young_kernel_ridge * np.eye(len(d))
        mature_matrix = kernel + config.mature_kernel_ridge * np.eye(len(d))
        coefficients_young = solve(young_matrix, target_young, assume_a="pos", check_finite=False)
        coefficients_mature = solve(mature_matrix, target_mature, assume_a="pos", check_finite=False)
        return cls(
            transform=transform, train_features=x,
            coefficients_young=coefficients_young, coefficients_mature=coefficients_mature,
            gammas=config.kernel_gammas, kernel_weights=config.kernel_weights,
            feature_names=FEATURE_NAMES,
        )

    def predict(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self.transform.transform(d)
        kernel = self.kernel(x, self.train_features, self.gammas, self.kernel_weights)
        return kernel @ self.coefficients_young, kernel @ self.coefficients_mature

    def mathematical_audit(self) -> dict[str, float]:
        coefficient_norm_young = float(np.linalg.norm(self.coefficients_young, ord=1))
        coefficient_norm_mature = float(np.linalg.norm(self.coefficients_mature, ord=1))
        maximum_gamma = max(self.gammas)
        feature_norm = float(np.linalg.norm(FEATURE_WEIGHTS))
        lipschitz_upper_young = coefficient_norm_young * maximum_gamma * math.sqrt(2 / math.e) * feature_norm
        lipschitz_upper_mature = coefficient_norm_mature * maximum_gamma * math.sqrt(2 / math.e) * feature_norm
        return {
            "kernel_coefficient_L1_young": coefficient_norm_young,
            "kernel_coefficient_L1_mature": coefficient_norm_mature,
            "finite_Lipschitz_upper_young": float(lipschitz_upper_young),
            "finite_Lipschitz_upper_mature": float(lipschitz_upper_mature),
        }


@dataclass
class HybridUSDE:
    mechanism: MechanisticModel
    closure: MultiScaleKernelClosure
    mechanistic_gate: float
    teacher_audit: dict[str, Any]

    @classmethod
    def fit(cls, d: pd.DataFrame, config: Configuration, seed: int) -> "HybridUSDE":
        mechanism = MechanisticModel.fit(d, config.integration_steps)
        teacher_config = TeacherConfiguration(
            young_trees=config.teacher_young_trees,
            mature_trees=config.teacher_mature_trees,
            joint_trees=config.teacher_joint_trees,
        )
        teacher_u, teacher_v, teacher_audit = cross_fitted_teacher_targets(d, teacher_config, seed)
        mechanism_u, mechanism_v = mechanism.predict(d)
        actual_delta_u = d.u1_log.to_numpy(float) - d.u0_log.to_numpy(float)
        actual_delta_v = d.v1_log.to_numpy(float) - d.v0_log.to_numpy(float)
        teacher_delta_u = teacher_u - d.u0_log.to_numpy(float)
        teacher_delta_v = teacher_v - d.v0_log.to_numpy(float)
        mechanism_delta_u = mechanism_u - d.u0_log.to_numpy(float)
        mechanism_delta_v = mechanism_v - d.v0_log.to_numpy(float)
        weight = config.distillation_data_weight
        closure_target_u = (
            (1 - weight) * teacher_delta_u + weight * actual_delta_u
            - config.mechanistic_gate * mechanism_delta_u
        )
        closure_target_v = (
            (1 - weight) * teacher_delta_v + weight * actual_delta_v
            - config.mechanistic_gate * mechanism_delta_v
        )
        closure = MultiScaleKernelClosure.fit(d, closure_target_u, closure_target_v, config)
        return cls(
            mechanism=mechanism, closure=closure,
            mechanistic_gate=config.mechanistic_gate, teacher_audit=teacher_audit,
        )

    def components(self, d: pd.DataFrame) -> dict[str, np.ndarray]:
        mechanism_u, mechanism_v = self.mechanism.predict(d)
        closure_u, closure_v = self.closure.predict(d)
        base_u = d.u0_log.to_numpy(float)
        base_v = d.v0_log.to_numpy(float)
        mechanism_delta_u = mechanism_u - base_u
        mechanism_delta_v = mechanism_v - base_v
        raw_u = np.maximum(base_u + self.mechanistic_gate * mechanism_delta_u + closure_u, 0.0)
        raw_v = np.maximum(base_v + self.mechanistic_gate * mechanism_delta_v + closure_v, 0.0)
        return {
            "base_u": base_u, "base_v": base_v,
            "mechanism_u": mechanism_u, "mechanism_v": mechanism_v,
            "mechanism_delta_u": mechanism_delta_u, "mechanism_delta_v": mechanism_delta_v,
            "closure_u": closure_u, "closure_v": closure_v,
            "raw_u": raw_u, "raw_v": raw_v,
        }


def choose_anchor(y: np.ndarray, base: np.ndarray, raw: np.ndarray) -> float:
    correction = raw - base
    rmse0 = np.sqrt(np.mean((y - base) ** 2)) + 1e-12
    mae0 = np.mean(np.abs(y - base)) + 1e-12
    grid = np.array([1.0])
    score = []
    for anchor in grid:
        prediction = np.maximum(base + anchor * correction, 0.0)
        score.append(
            0.75 * np.sqrt(np.mean((y - prediction) ** 2)) / rmse0
            + 0.25 * np.mean(np.abs(y - prediction)) / mae0
        )
    return float(grid[int(np.argmin(score))])


def estimate_stochastic_layer(
    d: pd.DataFrame,
    pred_u: np.ndarray,
    pred_v: np.ndarray,
) -> dict[str, float]:
    residual_u = d.u1_log.to_numpy(float) - pred_u
    residual_v = d.v1_log.to_numpy(float) - pred_v
    dt = d.dt_years.to_numpy(float)
    obs_u = 1.0 / (d.young_count_t1.to_numpy(float) + 0.5)
    obs_v = 1.0 / (d.mature_count_t1.to_numpy(float) + 0.5)

    def estimate(residual: np.ndarray, observation_variance: np.ndarray) -> tuple[float, float]:
        squared = residual * residual
        threshold = np.quantile(squared, 0.975)
        keep = np.isfinite(squared) & (squared <= threshold)
        design = np.column_stack([dt[keep], observation_variance[keep]])
        coefficients, _ = nnls(design, squared[keep])
        process_sigma = float(np.clip(math.sqrt(max(coefficients[0], 1e-8)), 0.005, 1.5))
        observation_scale = float(np.clip(math.sqrt(max(coefficients[1], 1e-8)), 0.01, 5.0))
        return process_sigma, observation_scale

    sigma_u, observation_u = estimate(residual_u, obs_u)
    sigma_v, observation_v = estimate(residual_v, obs_v)
    standardized_u = residual_u / np.sqrt(np.maximum(sigma_u * sigma_u * dt + observation_u * observation_u * obs_u, 1e-8))
    standardized_v = residual_v / np.sqrt(np.maximum(sigma_v * sigma_v * dt + observation_v * observation_v * obs_v, 1e-8))
    correlation = np.corrcoef(np.clip(standardized_u, -4, 4), np.clip(standardized_v, -4, 4))[0, 1]
    if not np.isfinite(correlation):
        correlation = 0.0
    correlation = float(np.clip(correlation, -0.90, 0.90))
    return {
        "sigma_young": sigma_u,
        "sigma_mature": sigma_v,
        "observation_scale_young": observation_u,
        "observation_scale_mature": observation_v,
        "brownian_correlation": correlation,
    }


def simulate_intervals(
    d: pd.DataFrame,
    components: dict[str, np.ndarray],
    anchor_young: float,
    anchor_mature: float,
    stochastic: dict[str, float],
    paths: int,
    seed: int,
    interval: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(d)
    dt = d.dt_years.to_numpy(float)
    root_dt = np.sqrt(dt)
    z1 = rng.normal(size=(paths, n))
    z2_independent = rng.normal(size=(paths, n))
    correlation = stochastic["brownian_correlation"]
    z2 = correlation * z1 + math.sqrt(max(1 - correlation * correlation, 1e-8)) * z2_independent
    expected_count_u = np.maximum(np.expm1(components["raw_u"]) * d.area_young_m2.to_numpy(float) / 10.0, 0.0)
    expected_count_v = np.maximum(np.expm1(components["raw_v"]) * d.area_mature_m2.to_numpy(float) / 10.0, 0.0)
    observation_sd_u = stochastic["observation_scale_young"] / np.sqrt(expected_count_u + 0.5)
    observation_sd_v = stochastic["observation_scale_mature"] / np.sqrt(expected_count_v + 0.5)
    observation_u = rng.normal(size=(paths, n)) * observation_sd_u
    observation_v = rng.normal(size=(paths, n)) * observation_sd_v
    path_u = components["raw_u"] + stochastic["sigma_young"] * root_dt * z1 + observation_u
    path_v = components["raw_v"] + stochastic["sigma_mature"] * root_dt * z2 + observation_v
    path_u = components["base_u"] + anchor_young * (path_u - components["base_u"])
    path_v = components["base_v"] + anchor_mature * (path_v - components["base_v"])
    path_u = np.maximum(path_u, 0.0)
    path_v = np.maximum(path_v, 0.0)
    tail = (1 - interval) / 2
    return {
        "pred_u": np.maximum(components["base_u"] + anchor_young * (components["raw_u"] - components["base_u"]), 0.0),
        "pred_v": np.maximum(components["base_v"] + anchor_mature * (components["raw_v"] - components["base_v"]), 0.0),
        "lower_u": np.quantile(path_u, tail, axis=0),
        "upper_u": np.quantile(path_u, 1 - tail, axis=0),
        "lower_v": np.quantile(path_v, tail, axis=0),
        "upper_v": np.quantile(path_v, 1 - tail, axis=0),
        "minimum_simulated_log_state": float(min(path_u.min(), path_v.min())),
        "maximum_simulated_log_state": float(max(path_u.max(), path_v.max())),
        "all_paths_finite": bool(np.isfinite(path_u).all() and np.isfinite(path_v).all()),
    }


def conformal_expansion(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, interval: float) -> float:
    score = np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])
    return finite_quantile(score, interval)


def fit_calibration(
    core: pd.DataFrame,
    calibration: pd.DataFrame,
    config: Configuration,
    seed: int,
) -> tuple[dict[str, Any], HybridUSDE]:
    model = HybridUSDE.fit(core, config, seed)
    components = model.components(calibration)
    anchor_young = choose_anchor(calibration.u1_log.to_numpy(float), components["base_u"], components["raw_u"])
    anchor_mature = choose_anchor(calibration.v1_log.to_numpy(float), components["base_v"], components["raw_v"])
    anchored_u = np.maximum(components["base_u"] + anchor_young * (components["raw_u"] - components["base_u"]), 0.0)
    anchored_v = np.maximum(components["base_v"] + anchor_mature * (components["raw_v"] - components["base_v"]), 0.0)
    stochastic = estimate_stochastic_layer(calibration, anchored_u, anchored_v)
    simulation = simulate_intervals(
        calibration, components, anchor_young, anchor_mature, stochastic,
        config.mc_paths, seed + 700001, config.interval,
    )
    calibration_probability = min(0.98, config.interval + config.conformal_site_shift_margin)
    q_young = conformal_expansion(
        calibration.u1_log.to_numpy(float), simulation["lower_u"], simulation["upper_u"], calibration_probability,
    )
    q_mature = conformal_expansion(
        calibration.v1_log.to_numpy(float), simulation["lower_v"], simulation["upper_v"], calibration_probability,
    )
    calibration_output = {
        "anchor_young": anchor_young,
        "anchor_mature": anchor_mature,
        "conformal_q_young": q_young,
        "conformal_q_mature": q_mature,
        **stochastic,
        "calibration_sites": sorted(calibration.siteID.astype(str).unique().tolist()),
    }
    return calibration_output, model


def predict_with_calibration(
    model: HybridUSDE,
    d: pd.DataFrame,
    calibration: dict[str, Any],
    config: Configuration,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    components = model.components(d)
    simulation = simulate_intervals(
        d, components,
        calibration["anchor_young"], calibration["anchor_mature"], calibration,
        config.mc_paths, seed, config.interval,
    )
    lower_u = np.maximum(simulation["lower_u"] - calibration["conformal_q_young"], 0.0)
    upper_u = simulation["upper_u"] + calibration["conformal_q_young"]
    lower_v = np.maximum(simulation["lower_v"] - calibration["conformal_q_mature"], 0.0)
    upper_v = simulation["upper_v"] + calibration["conformal_q_mature"]
    output = pd.DataFrame({
        "pred_u_log": simulation["pred_u"], "lower_u_log": lower_u, "upper_u_log": upper_u,
        "pred_v_log": simulation["pred_v"], "lower_v_log": lower_v, "upper_v_log": upper_v,
        "mechanism_u_log": components["mechanism_u"], "mechanism_v_log": components["mechanism_v"],
        "closure_u_log": components["closure_u"], "closure_v_log": components["closure_v"],
        "raw_u_log": components["raw_u"], "raw_v_log": components["raw_v"],
    })
    audit = {
        "minimum_simulated_log_state": simulation["minimum_simulated_log_state"],
        "maximum_simulated_log_state": simulation["maximum_simulated_log_state"],
        "all_paths_finite": simulation["all_paths_finite"],
    }
    return output, audit


def split_core_calibration(d: pd.DataFrame, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(d))
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    core, calibration = next(splitter.split(indices, groups=d.siteID.to_numpy()))
    return core, calibration


def run_outer_cv(d: pd.DataFrame, config: Configuration) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(d))
    groups = d.siteID.to_numpy()
    predictions = []
    audits = []
    parameter_tables = []
    calibration_tables = []
    for fold, (train, test) in enumerate(GroupKFold(config.folds).split(indices, groups=groups), start=1):
        outer_train = d.iloc[train].reset_index(drop=True)
        core_rel, calibration_rel = split_core_calibration(outer_train, config.calibration_fraction, config.seed + fold * 101)
        core = outer_train.iloc[core_rel].reset_index(drop=True)
        calibration_frame = outer_train.iloc[calibration_rel].reset_index(drop=True)
        calibration, _ = fit_calibration(core, calibration_frame, config, config.seed + fold * 100000)
        final_model = HybridUSDE.fit(outer_train, config, config.seed + fold * 100000 + 50000)
        fold_prediction, path_audit = predict_with_calibration(
            final_model, d.iloc[test].reset_index(drop=True), calibration, config,
            config.seed + fold * 100000 + 90000,
        )
        fold_prediction.insert(0, "row_id", test)
        fold_prediction.insert(1, "fold", fold)
        predictions.append(fold_prediction)
        train_sites = set(d.iloc[train].siteID.astype(str))
        test_sites = set(d.iloc[test].siteID.astype(str))
        audits.append({
            "fold": fold, "train_rows": len(train), "test_rows": len(test),
            "train_sites": len(train_sites), "test_sites": len(test_sites),
            "site_overlap": len(train_sites & test_sites),
            "core_sites": core.siteID.nunique(), "calibration_sites": calibration_frame.siteID.nunique(),
            "calibration_site_overlap_core": len(set(core.siteID) & set(calibration_frame.siteID)),
            **path_audit,
            **final_model.closure.mathematical_audit(),
        })
        parameter_tables.append(final_model.mechanism.parameter_table(f"fold_{fold}"))
        calibration_tables.append(pd.DataFrame({"fold": fold, "item": list(calibration.keys()), "value": [json.dumps(value) if isinstance(value, list) else value for value in calibration.values()]}))
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(audits),
        pd.concat(parameter_tables, ignore_index=True),
        pd.concat(calibration_tables, ignore_index=True),
    )


def build_oof(d: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transition_id", "siteID", "plotID", "dt_years", "event_year",
        "decimalLatitude", "decimalLongitude", "area_young_m2", "area_mature_m2",
        "young_count_t0", "mature_count_t0", "young_count_t1", "mature_count_t1",
        "u0_kha", "v0_kha", "u1_kha", "v1_kha", "u0_log", "v0_log", "u1_log", "v1_log",
    ]
    oof = d[columns].copy()
    oof["row_id"] = np.arange(len(d))
    oof = oof.merge(prediction, on="row_id", how="left")
    oof["persistence_u_log"] = oof.u0_log
    oof["persistence_v_log"] = oof.v0_log
    for key in ["u", "v"]:
        for prefix in ["pred", "lower", "upper", "mechanism", "raw"]:
            oof[f"{prefix}_{key}_kha"] = np.expm1(oof[f"{prefix}_{key}_log"])
        oof[f"annual_closure_{key}_log"] = oof[f"closure_{key}_log"] / oof.dt_years
    return oof.sort_values("row_id").reset_index(drop=True)


def safe_spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    if np.unique(y).size < 2 or np.unique(prediction).size < 2:
        return float("nan")
    return float(spearmanr(y, prediction).statistic)


def metric_values(y: np.ndarray, prediction: np.ndarray, lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "RMSE_log1p": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE_log1p": float(mean_absolute_error(y, prediction)),
        "R2_log1p": float(r2_score(y, prediction)),
        "Spearman": safe_spearman(y, prediction),
        "Bias_log1p": float(np.mean(prediction - y)),
        "RMSE_kha": float(np.sqrt(mean_squared_error(np.expm1(y), np.expm1(prediction)))),
        "MAE_kha": float(mean_absolute_error(np.expm1(y), np.expm1(prediction))),
    }
    if lower is not None and upper is not None:
        result["Coverage"] = float(np.mean((y >= lower) & (y <= upper)))
        result["Mean_interval_width_log1p"] = float(np.mean(upper - lower))
    return result


def bootstrap_metrics(oof: pd.DataFrame, stage: str, repetitions: int, seed: int) -> pd.DataFrame:
    key = "u" if stage == "young" else "v"
    y = oof[f"{key}1_log"].to_numpy(float)
    model = oof[f"pred_{key}_log"].to_numpy(float)
    persistence = oof[f"persistence_{key}_log"].to_numpy(float)
    mechanism = oof[f"mechanism_{key}_log"].to_numpy(float)
    smooth_only = np.maximum(persistence + oof[f"closure_{key}_log"].to_numpy(float), 0.0)
    lower = oof[f"lower_{key}_log"].to_numpy(float)
    upper = oof[f"upper_{key}_log"].to_numpy(float)
    point_models = {
        "PACT-USDE": metric_values(y, model, lower, upper),
        "Persistence": metric_values(y, persistence),
        "Mechanistic ODE skeleton": metric_values(y, mechanism),
        "Smooth closure only": metric_values(y, smooth_only),
    }
    sites = oof.siteID.astype(str).to_numpy()
    unique_sites = np.unique(sites)
    site_rows = {site: np.flatnonzero(sites == site) for site in unique_sites}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in point_models["PACT-USDE"]}
    skills = []
    for _ in range(repetitions):
        sampled_sites = rng.choice(unique_sites, len(unique_sites), replace=True)
        sampled = np.concatenate([site_rows[site] for site in sampled_sites])
        current = metric_values(y[sampled], model[sampled], lower[sampled], upper[sampled])
        benchmark = metric_values(y[sampled], persistence[sampled])
        for metric, value in current.items():
            draws[metric].append(value)
        skills.append(1 - current["RMSE_log1p"] / benchmark["RMSE_log1p"])
    rows = []
    for model_name, values in point_models.items():
        for metric, estimate in values.items():
            if model_name == "PACT-USDE":
                distribution = np.asarray(draws[metric], dtype=float)
                lower_ci = float(np.nanquantile(distribution, 0.025))
                upper_ci = float(np.nanquantile(distribution, 0.975))
            else:
                lower_ci = upper_ci = float("nan")
            rows.append({
                "stage": stage, "model": model_name, "metric": metric,
                "estimate": estimate, "ci_lower": lower_ci, "ci_upper": upper_ci,
            })
    skill = 1 - point_models["PACT-USDE"]["RMSE_log1p"] / point_models["Persistence"]["RMSE_log1p"]
    rows.append({
        "stage": stage, "model": "PACT-USDE vs Persistence", "metric": "RMSE_skill",
        "estimate": skill, "ci_lower": float(np.quantile(skills, 0.025)),
        "ci_upper": float(np.quantile(skills, 0.975)),
    })
    return pd.DataFrame(rows)


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, frame in oof.groupby("fold"):
        for stage, key in [("young", "u"), ("mature", "v")]:
            y = frame[f"{key}1_log"].to_numpy(float)
            prediction = frame[f"pred_{key}_log"].to_numpy(float)
            persistence = frame[f"persistence_{key}_log"].to_numpy(float)
            values = metric_values(y, prediction, frame[f"lower_{key}_log"], frame[f"upper_{key}_log"])
            persistence_rmse = np.sqrt(mean_squared_error(y, persistence))
            rows.append({
                "fold": fold, "stage": stage, **values,
                "persistence_RMSE_log1p": persistence_rmse,
                "RMSE_skill": 1 - values["RMSE_log1p"] / persistence_rmse,
            })
    return pd.DataFrame(rows)


def grouped_skill(oof: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for name, frame in oof.groupby(group, observed=True):
        if len(frame) < 3:
            continue
        for stage, key in [("young", "u"), ("mature", "v")]:
            y = frame[f"{key}1_log"].to_numpy(float)
            prediction = frame[f"pred_{key}_log"].to_numpy(float)
            persistence = frame[f"persistence_{key}_log"].to_numpy(float)
            rmse_model = np.sqrt(mean_squared_error(y, prediction))
            rmse_persistence = np.sqrt(mean_squared_error(y, persistence))
            rows.append({
                group: name, "stage": stage, "n": len(frame),
                "RMSE_model": rmse_model, "RMSE_persistence": rmse_persistence,
                "absolute_RMSE_gain": rmse_persistence - rmse_model,
                "RMSE_skill": 1 - rmse_model / rmse_persistence if rmse_persistence > 0 else np.nan,
                "MAE_model": mean_absolute_error(y, prediction), "R2_model": r2_score(y, prediction),
                "Coverage": np.mean((y >= frame[f"lower_{key}_log"]) & (y <= frame[f"upper_{key}_log"])),
            })
    return pd.DataFrame(rows)


def horizon_skill(oof: pd.DataFrame) -> pd.DataFrame:
    frame = oof.copy()
    frame["horizon_bin"] = pd.qcut(frame.dt_years, q=5, duplicates="drop").astype(str)
    result = grouped_skill(frame, "horizon_bin")
    ranges = frame.groupby("horizon_bin", observed=True).dt_years.agg(["min", "median", "max"]).reset_index()
    site_counts = frame.groupby("horizon_bin", observed=True).siteID.nunique().rename("n_sites").reset_index()
    return result.merge(ranges, on="horizon_bin", how="left").merge(site_counts, on="horizon_bin", how="left")


def climate_loadings(model: HybridUSDE) -> pd.DataFrame:
    pca = model.closure.transform.climate.pca
    rows = []
    for component in range(2):
        for feature, loading in zip(RAW_CLIMATE, pca.components_[component]):
            rows.append({
                "component": f"PC{component + 1}", "feature": feature, "loading": loading,
                "explained_variance_ratio": pca.explained_variance_ratio_[component],
                "cumulative_explained_variance": pca.explained_variance_ratio_[: component + 1].sum(),
            })
    return pd.DataFrame(rows)


def climate_rate_response(model: HybridUSDE) -> pd.DataFrame:
    grid = np.linspace(-2.5, 2.5, 101)
    rows = []
    for varied in [0, 1]:
        climate = np.zeros((len(grid), 2))
        climate[:, varied] = grid
        q, maturation, mortality, rho, mu, a, b = MechanisticModel.rates(model.mechanism.parameters, climate)
        for index, value in enumerate(grid):
            rows.append({
                "varied_component": f"PC{varied + 1}", "component_value": value,
                "recruitment_q": q[index], "maturation_F": maturation[index],
                "mature_loss_H": mortality[index], "rho": rho, "mu": mu,
                "young_density_regulation_a": a, "mature_density_regulation_b": b,
            })
    return pd.DataFrame(rows)


def theory_table(model: HybridUSDE) -> pd.DataFrame:
    audit = model.closure.mathematical_audit()
    rows = [
        ("State space", "The latent process evolves in the positive orthant (0, infinity)^2."),
        ("Local regularity", "Logistic climate rates and Gaussian RBF closure terms are continuously differentiable and locally Lipschitz."),
        ("Boundary behavior", "Recruitment and maturation inflows are nonnegative, while multiplicative diffusion vanishes at the boundary."),
        ("Dissipativity", "Negative quadratic density-regulation terms -a U^2 and -b V^2 dominate bounded closure terms at large states."),
        ("Global solution", "A stopping-time argument with V(U,V)=U+V-log(U)-log(V) yields positivity and non-explosion, hence a unique global strong solution."),
        ("Closure boundedness", f"The finite RBF coefficient norms imply finite global closure bounds; audited L1 norms are {audit['kernel_coefficient_L1_young']:.3f} and {audit['kernel_coefficient_L1_mature']:.3f}."),
        ("Closure Lipschitz audit", f"Finite computable Lipschitz upper bounds are {audit['finite_Lipschitz_upper_young']:.3f} and {audit['finite_Lipschitz_upper_mature']:.3f}."),
        ("Inference scope", "The theorem concerns the fitted continuous-time latent process; predictive associations are not interpreted as causal effects."),
    ]
    return pd.DataFrame({"item": [row[0] for row in rows], "statement": [row[1] for row in rows]})


def make_figures(oof: pd.DataFrame, horizon: pd.DataFrame, rates: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for axis, stage, key in zip(axes, ["Young", "Mature"], ["u", "v"]):
        observed = oof[f"{key}1_log"].to_numpy(float)
        prediction = oof[f"pred_{key}_log"].to_numpy(float)
        persistence = oof[f"persistence_{key}_log"].to_numpy(float)
        limit = max(observed.max(), prediction.max(), persistence.max()) * 1.02
        axis.scatter(observed, persistence, s=9, alpha=0.12, label="Persistence")
        axis.scatter(observed, prediction, s=11, alpha=0.30, label="PACT-USDE")
        axis.plot([0, limit], [0, limit], "--", linewidth=1.2, label="1:1")
        axis.set(xlim=(0, limit), ylim=(0, limit), xlabel=f"Observed log1p {stage.lower()} density", ylabel=f"Predicted log1p {stage.lower()} density", title=stage)
    axes[0].legend(frameon=False)
    fig.suptitle("Site-blocked out-of-fold prediction")
    fig.tight_layout()
    fig.savefig(out / "figure_1_prediction.png", dpi=dpi)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for stage, frame in horizon.groupby("stage"):
        ordered = frame.sort_values("median")
        axis.plot(ordered["median"], 100 * ordered.RMSE_skill, marker="o", label=stage.capitalize())
    axis.axhline(0, linestyle="--", linewidth=1)
    axis.set(xlabel="Median census interval (years)", ylabel="RMSE skill relative to persistence (%)", title="Benchmark-relative skill across observed census intervals")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure_2_horizon_skill.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, component in zip(axes, ["PC1", "PC2"]):
        frame = rates[rates.varied_component == component]
        axis.plot(frame.component_value, frame.recruitment_q, label="Recruitment q")
        axis.plot(frame.component_value, frame.maturation_F, label="Maturation F")
        axis.plot(frame.component_value, frame.mature_loss_H, label="Mature loss H")
        axis.set(xlabel=f"{component} score", ylabel="Annual bounded rate", title=f"Mechanistic climate response along {component}")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure_3_climate_rates.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, stage, key in zip(axes, ["Young", "Mature"], ["u", "v"]):
        mechanism = oof[f"mechanism_{key}_log"] - oof[f"persistence_{key}_log"]
        closure = oof[f"closure_{key}_log"]
        axis.boxplot([mechanism, closure], tick_labels=["Mechanistic change", "Smooth closure"])
        axis.axhline(0, linestyle="--", linewidth=1)
        axis.set(ylabel="Log1p transition contribution", title=stage)
    fig.suptitle("Decomposition of the fitted transition operator")
    fig.tight_layout()
    fig.savefig(out / "figure_4_component_decomposition.png", dpi=dpi)
    plt.close(fig)


def save_tables_as_csv(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--out", default="results_pact_usde")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--mc-paths", type=int, default=128)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.folds != 3:
        raise ValueError("PACT-USDE is audited for exactly three site-blocked outer folds")
    start = time.time()
    root = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    config = Configuration(
        folds=args.folds, bootstrap=args.bootstrap, mc_paths=args.mc_paths,
        integration_steps=args.steps, seed=args.seed, dpi=args.dpi,
    )
    data = prepare_data(data_path)
    print(f"Data: {len(data)} transitions, {data.plotID.nunique()} plots, {data.siteID.nunique()} sites")
    print("Running three-fold site-blocked evaluation...")
    oof_prediction, cv_audit, fold_parameters, calibration_audit = run_outer_cv(data, config)
    oof = build_oof(data, oof_prediction)
    metrics = pd.concat([
        bootstrap_metrics(oof, "young", config.bootstrap, config.seed + 1),
        bootstrap_metrics(oof, "mature", config.bootstrap, config.seed + 2),
    ], ignore_index=True)
    folds = fold_metrics(oof)
    sites = grouped_skill(oof, "siteID")
    horizon = horizon_skill(oof)

    print("Fitting final deployment model...")
    core_rel, calibration_rel = split_core_calibration(data, config.calibration_fraction, config.seed + 777)
    final_calibration, _ = fit_calibration(
        data.iloc[core_rel].reset_index(drop=True), data.iloc[calibration_rel].reset_index(drop=True),
        config, config.seed + 800000,
    )
    final_model = HybridUSDE.fit(data, config, config.seed + 900000)
    final_prediction, final_path_audit = predict_with_calibration(
        final_model, data.reset_index(drop=True), final_calibration, config, config.seed + 950000,
    )
    final = data[[
        "transition_id", "siteID", "plotID", "dt_years", "u0_kha", "v0_kha", "u1_kha", "v1_kha"
    ]].copy().reset_index(drop=True)
    final = pd.concat([final, final_prediction], axis=1)
    for key in ["u", "v"]:
        for prefix in ["pred", "lower", "upper", "mechanism", "raw"]:
            final[f"{prefix}_{key}_kha"] = np.expm1(final[f"{prefix}_{key}_log"])

    pca = climate_loadings(final_model)
    rates = climate_rate_response(final_model)
    theory = theory_table(final_model)
    final_parameters = final_model.mechanism.parameter_table("final")
    raw_correlation = data[RAW_CLIMATE].corr().stack().reset_index()
    raw_correlation.columns = ["variable_1", "variable_2", "correlation"]
    raw_correlation = raw_correlation[raw_correlation.variable_1 < raw_correlation.variable_2].reset_index(drop=True)
    step_u_12, step_v_12 = final_model.mechanism.predict(data, steps=config.integration_steps)
    step_u_24, step_v_24 = final_model.mechanism.predict(data, steps=2 * config.integration_steps)
    numerical_audit = pd.DataFrame({
        "item": [
            "all_outer_simulated_paths_finite", "all_outer_site_overlaps_zero",
            "all_outer_calibration_overlaps_zero", "minimum_outer_simulated_log_state",
            "maximum_outer_simulated_log_state", "final_all_paths_finite",
            "final_minimum_simulated_log_state", "final_maximum_simulated_log_state",
            "mean_step_sensitivity_young_log", "max_step_sensitivity_young_log",
            "mean_step_sensitivity_mature_log", "max_step_sensitivity_mature_log",
        ],
        "value": [
            bool(cv_audit.all_paths_finite.all()), bool((cv_audit.site_overlap == 0).all()),
            bool((cv_audit.calibration_site_overlap_core == 0).all()), cv_audit.minimum_simulated_log_state.min(),
            cv_audit.maximum_simulated_log_state.max(), final_path_audit["all_paths_finite"],
            final_path_audit["minimum_simulated_log_state"], final_path_audit["maximum_simulated_log_state"],
            np.mean(np.abs(step_u_12 - step_u_24)), np.max(np.abs(step_u_12 - step_u_24)),
            np.mean(np.abs(step_v_12 - step_v_24)), np.max(np.abs(step_v_12 - step_v_24)),
        ],
    })
    summary_items = [
        "model", "transitions", "plots", "sites", "outer_folds", "runtime_seconds",
        "young_RMSE", "young_MAE", "young_R2", "young_Spearman", "young_RMSE_skill", "young_Coverage",
        "mature_RMSE", "mature_MAE", "mature_R2", "mature_Spearman", "mature_RMSE_skill", "mature_Coverage",
    ]

    def metric(stage: str, model: str, name: str) -> float:
        return float(metrics.query("stage == @stage and model == @model and metric == @name").estimate.iloc[0])

    summary_values = [
        MODEL_NAME, len(data), data.plotID.nunique(), data.siteID.nunique(), config.folds, time.time() - start,
        metric("young", "PACT-USDE", "RMSE_log1p"), metric("young", "PACT-USDE", "MAE_log1p"),
        metric("young", "PACT-USDE", "R2_log1p"), metric("young", "PACT-USDE", "Spearman"),
        metric("young", "PACT-USDE vs Persistence", "RMSE_skill"), metric("young", "PACT-USDE", "Coverage"),
        metric("mature", "PACT-USDE", "RMSE_log1p"), metric("mature", "PACT-USDE", "MAE_log1p"),
        metric("mature", "PACT-USDE", "R2_log1p"), metric("mature", "PACT-USDE", "Spearman"),
        metric("mature", "PACT-USDE vs Persistence", "RMSE_skill"), metric("mature", "PACT-USDE", "Coverage"),
    ]
    summary = pd.DataFrame({"metric": summary_items, "value": summary_values})
    run_audit = pd.DataFrame({
        "item": [
            "model", "data", "data_sha256", "python", "numpy", "pandas", "scipy", "scikit_learn",
            "platform", "seed", "folds", "bootstrap", "mc_paths", "integration_steps",
            "distillation_data_weight", "mechanistic_gate", "kernel_gammas", "kernel_weights",
            "young_kernel_ridge", "mature_kernel_ridge", "conformal_site_shift_margin", "teacher_audit", "final_calibration",
        ],
        "value": [
            MODEL_NAME, str(data_path.resolve()), sha256(data_path), sys.version, np.__version__, pd.__version__,
            scipy.__version__, sklearn.__version__, platform.platform(), config.seed, config.folds,
            config.bootstrap, config.mc_paths, config.integration_steps, config.distillation_data_weight,
            config.mechanistic_gate, json.dumps(config.kernel_gammas), json.dumps(config.kernel_weights),
            config.young_kernel_ridge, config.mature_kernel_ridge, config.conformal_site_shift_margin, json.dumps(final_model.teacher_audit),
            json.dumps(final_calibration),
        ],
    })
    tables = {
        "Summary": summary, "CV_Metrics": metrics, "Fold_Metrics": folds,
        "OOF_Predictions": oof, "Final_Predictions": final,
        "Site_Skill": sites, "Horizon_Skill": horizon,
        "CV_Audit": cv_audit, "Calibration_Audit": calibration_audit,
        "Fold_Parameters": fold_parameters, "Final_Parameters": final_parameters,
        "PCA_Loadings": pca, "Raw_Climate_Correlation": raw_correlation,
        "Climate_Rate_Response": rates, "Mathematical_Theory": theory,
        "Numerical_Audit": numerical_audit, "Run_Audit": run_audit,
    }
    save_tables_as_csv(out, tables)
    portable_model_state = {
        "mechanism": {
            "climate_scaler": final_model.mechanism.climate.scaler,
            "climate_pca": final_model.mechanism.climate.pca,
            "parameters": final_model.mechanism.parameters,
            "integration_steps": final_model.mechanism.integration_steps,
        },
        "closure": {
            "feature_climate_scaler": final_model.closure.transform.climate.scaler,
            "feature_climate_pca": final_model.closure.transform.climate.pca,
            "feature_scaler": final_model.closure.transform.scaler,
            "train_features": final_model.closure.train_features,
            "coefficients_young": final_model.closure.coefficients_young,
            "coefficients_mature": final_model.closure.coefficients_mature,
            "gammas": final_model.closure.gammas,
            "kernel_weights": final_model.closure.kernel_weights,
            "feature_weights": FEATURE_WEIGHTS,
            "feature_names": FEATURE_NAMES,
        },
        "mechanistic_gate": final_model.mechanistic_gate,
        "teacher_audit": final_model.teacher_audit,
    }
    joblib.dump({
        "model_name": MODEL_NAME, "model_state": portable_model_state,
        "calibration": final_calibration, "configuration": config.__dict__,
    }, out / "pact_usde_model.joblib", compress=3)
    manifest = {
        "model": MODEL_NAME, "data": str(data_path.resolve()), "data_sha256": sha256(data_path),
        "configuration": config.__dict__, "runtime_seconds": time.time() - start,
        "teacher_audit": final_model.teacher_audit, "calibration": final_calibration,
        "mathematical_audit": final_model.closure.mathematical_audit(),
        "numerical_audit": dict(zip(numerical_audit.item, numerical_audit.value)),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    make_figures(oof, horizon, rates, out, config.dpi)
    print(summary.to_string(index=False))
    print(f"Completed: {out}")
    print(f"Runtime: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()