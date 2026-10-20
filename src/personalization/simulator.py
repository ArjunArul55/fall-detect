"""Deterministic, pluggable shifts for simulating out-of-distribution users."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class UserProfile:
    name: str
    amplitude_scale: float = 1.0
    rotation_degrees: float = 0.0
    noise_std: float = 0.0
    speed_range: tuple[float, float] = (1.0, 1.0)
    bias_std: float = 0.0


USER_PROFILES: dict[str, UserProfile] = {
    "Normal": UserProfile("Normal"),
    "Low-amp": UserProfile("Low-amp", amplitude_scale=0.6),
    "High-amp": UserProfile("High-amp", amplitude_scale=1.4),
    "Rotated": UserProfile("Rotated", rotation_degrees=35.0, bias_std=0.04),
    "Noisy": UserProfile("Noisy", noise_std=0.08, bias_std=0.03),
    "Slow": UserProfile("Slow", speed_range=(0.8, 0.95), bias_std=0.02),
}


class UserSimulator:
    """Apply a seeded parametric user shift or a pluggable window generator."""

    def __init__(
        self,
        backend: Literal["parametric", "generator"] = "parametric",
        *,
        seed: int = 42,
        generator: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
    ) -> None:
        if backend not in ("parametric", "generator"):
            raise ValueError(f"Unsupported user simulator backend: {backend!r}.")
        if backend == "generator" and generator is None:
            raise ValueError("The generator backend requires a fitted window generator.")
        self.backend = backend
        self.seed = int(seed)
        self.generator = generator

    @staticmethod
    def _rotation_matrix(degrees: float) -> np.ndarray:
        angle = np.deg2rad(degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _user_bias(self, profile: UserProfile, user_id: str) -> np.ndarray:
        stable_id = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, stable_id]))
        return rng.normal(0.0, profile.bias_std, size=6).astype(np.float32)

    @staticmethod
    def _time_warp(window: np.ndarray, speed: float) -> np.ndarray:
        time = np.arange(window.shape[0], dtype=np.float32)
        center = (window.shape[0] - 1) / 2.0
        source = center + (time - center) * speed
        source = np.clip(source, 0, window.shape[0] - 1)
        warped = np.stack(
            [np.interp(source, time, window[:, channel]) for channel in range(window.shape[1])],
            axis=1,
        )
        return warped.astype(np.float32, copy=False)

    def transform(
        self,
        windows: np.ndarray,
        profile: str | UserProfile,
        *,
        user_id: str = "user",
        seed: int | None = None,
    ) -> np.ndarray:
        """Transform `(N, 64, 6)` windows without changing their labels/order."""

        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3 or tuple(values.shape[1:]) != (64, 6):
            raise ValueError(f"windows must have shape (N, 64, 6), got {values.shape}.")
        if self.backend == "generator":
            raise ValueError(
                "A generative backend creates labeled windows; use generate(labels) "
                "instead of transforming existing windows."
            )

        settings = USER_PROFILES[profile] if isinstance(profile, str) else profile
        rng = np.random.default_rng(self.seed if seed is None else seed)
        result = values.copy()
        rotation = self._rotation_matrix(settings.rotation_degrees)
        result[..., :3] = result[..., :3] @ rotation.T
        result[..., 3:] = result[..., 3:] @ rotation.T
        result *= np.float32(settings.amplitude_scale)
        bias = self._user_bias(settings, user_id)
        result += bias
        low, high = settings.speed_range
        for index in range(len(result)):
            if low != 1.0 or high != 1.0:
                result[index] = self._time_warp(result[index], float(rng.uniform(low, high)))
            if settings.noise_std:
                result[index] += rng.normal(0.0, settings.noise_std, size=result[index].shape).astype(np.float32)
        return result

    def generate(self, labels: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        """Generate class-conditional windows through the configured backend."""

        if self.backend != "generator" or self.generator is None:
            raise ValueError("generate(labels) is available only with the generator backend.")
        classes = np.asarray(labels, dtype=np.int32).reshape(-1)
        if not np.isin(classes, [0, 1]).all():
            raise ValueError("Generation labels must be binary.")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        result = np.asarray(self.generator(classes, rng), dtype=np.float32)
        if result.shape != (len(classes), 64, 6):
            raise ValueError(
                f"Generator returned shape {result.shape}; expected {(len(classes), 64, 6)}."
            )
        return result


def split_user_calibration(
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    max_calibration: int = 30,
    seed: int = 42,
    allow_adl_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reserve disjoint ADL-only calibration rows and leave ADL+falls for query."""

    x = np.asarray(windows)
    y = np.asarray(labels)
    if len(x) != len(y) or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("windows and binary labels must be aligned.")
    if max_calibration < 0:
        raise ValueError("max_calibration must be non-negative.")
    adl = np.flatnonzero(y == 0)
    falls = np.flatnonzero(y == 1)
    if not len(adl) or (not len(falls) and not allow_adl_only):
        raise ValueError("A user partition requires ADL windows and, unless allowed, fall windows.")
    rng = np.random.default_rng(seed)
    shuffled_adl = rng.permutation(adl)
    calibration_count = min(max_calibration, max(0, len(shuffled_adl) - 1))
    calibration_indices = shuffled_adl[:calibration_count]
    query_indices = np.concatenate((shuffled_adl[calibration_count:], falls))
    query_indices = rng.permutation(query_indices)
    return (
        x[calibration_indices],
        y[calibration_indices],
        x[query_indices],
        y[query_indices],
    )


def assert_subject_disjoint(
    train_subject_ids: np.ndarray,
    val_subject_ids: np.ndarray,
    test_subject_ids: np.ndarray,
    *,
    generator_subject_ids: np.ndarray | None = None,
) -> None:
    """Fail closed if window-level subject metadata leaks across partitions."""

    train = set(np.asarray(train_subject_ids).astype(str))
    validation = set(np.asarray(val_subject_ids).astype(str))
    test = set(np.asarray(test_subject_ids).astype(str))
    if train & validation or train & test or validation & test:
        raise ValueError("Subject leakage detected between train, validation, and test.")
    if generator_subject_ids is not None and set(np.asarray(generator_subject_ids).astype(str)) - train:
        raise ValueError("Generator training may use train-subject windows only.")
