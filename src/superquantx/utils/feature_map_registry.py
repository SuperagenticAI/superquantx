"""Feature-map name registry for SuperQuantX quantum kernels.

Backends historically accept a small set of string names
(``ZZFeatureMap``, ``PauliFeatureMap``, ``AmplitudeMap``) plus a default
angle-encoding fallback. This module is the single source of truth for
those names, plus aliases used by the Qmes-inspired auto selector.
"""

from __future__ import annotations

from typing import Final


# Canonical names that QuantumSVM / backends should recognize.
FEATURE_MAP_REGISTRY: Final[dict[str, str]] = {
    "ZZFeatureMap": (
        "Hadamard + RZ data encoding with linear ZZ (RZZ/CNOT) entanglement. "
        "Closest SuperQuantX analogue of Qmes HD / strongly entangling maps."
    ),
    "PauliFeatureMap": (
        "Per-qubit RX/RY/RZ encoding with linear CNOT entanglement. "
        "Closest analogue of Qmes HERx and RY_CX."
    ),
    "AmplitudeMap": (
        "Amplitude-style embedding (compact state preparation). "
        "Closest analogue of Qmes unit (square-root amplitude)."
    ),
    "AngleEncoding": (
        "Separable single-qubit RY (or RX) rotations without entanglement. "
        "Closest analogue of Qmes RY and SRx."
    ),
    "ZFeatureMap": (
        "Separable RZ rotations (no entanglement). "
        "Closest analogue of Qmes ZFM."
    ),
}

# Names accepted by create_feature_map on backends today (case-sensitive).
SUPPORTED_FEATURE_MAP_NAMES: Final[tuple[str, ...]] = tuple(FEATURE_MAP_REGISTRY.keys())

# Default when the caller does not request auto selection.
DEFAULT_FEATURE_MAP: Final[str] = "ZZFeatureMap"

# Qmes circuit pool -> SuperQuantX registry name.
# Gaps (exact gate schedules, qubit count fixed at 4 in Qmes, ISWAP brickwork
# for HD, etc.) are intentional for this spike; see docs/design/qmes-feature-map-auto.md.
QMES_TO_SUPERQUANTX: Final[dict[str, str]] = {
    "unit": "AmplitudeMap",
    "SRx": "AngleEncoding",
    "RY": "AngleEncoding",
    "HERx": "PauliFeatureMap",
    "RY_CX": "PauliFeatureMap",
    "ZFM": "ZFeatureMap",
    "HD": "ZZFeatureMap",
}


def normalize_feature_map_name(name: str) -> str:
    """Normalize aliases to a canonical registry name.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name in FEATURE_MAP_REGISTRY:
        return name

    aliases = {
        "zz": "ZZFeatureMap",
        "zzfeaturemap": "ZZFeatureMap",
        "pauli": "PauliFeatureMap",
        "paulifeaturemap": "PauliFeatureMap",
        "amplitude": "AmplitudeMap",
        "amplitudemap": "AmplitudeMap",
        "angle": "AngleEncoding",
        "angleencoding": "AngleEncoding",
        "ry": "AngleEncoding",
        "z": "ZFeatureMap",
        "zfeaturemap": "ZFeatureMap",
        "zfm": "ZFeatureMap",
    }
    key = name.replace("_", "").replace("-", "").lower()
    if key in aliases:
        return aliases[key]

    raise ValueError(
        f"Unknown feature map '{name}'. "
        f"Supported: {list(SUPPORTED_FEATURE_MAP_NAMES)}"
    )


def list_feature_maps() -> list[str]:
    """Return canonical feature-map names in registry order."""
    return list(SUPPORTED_FEATURE_MAP_NAMES)


def map_qmes_circuit(qmes_name: str) -> str:
    """Map a Qmes circuit-pool name to a SuperQuantX registry entry."""
    if qmes_name not in QMES_TO_SUPERQUANTX:
        raise ValueError(
            f"Unknown Qmes circuit '{qmes_name}'. "
            f"Known: {list(QMES_TO_SUPERQUANTX)}"
        )
    return QMES_TO_SUPERQUANTX[qmes_name]
