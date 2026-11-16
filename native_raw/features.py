"""Build Native model inputs in the order used during training."""

import numpy as np

from .base_features import make_features
from .temporal_features import build_extra


def apply_normalizer(values, validity, normalizer):
    """Standardize feature values and append their validity flags."""
    f = len(normalizer["feature_names"])
    if values.ndim != 2 or values.shape[1] != f or validity.shape != values.shape:
        raise ValueError("Feature shape differs from the normalizer")
    if not np.array_equal(validity, np.isfinite(values)):
        raise ValueError("Stored feature validity does not match finite values")
    shift = np.asarray(normalizer["shift"], dtype=np.float64)
    scale = np.asarray(normalizer["scale"], dtype=np.float64)
    active = np.asarray(normalizer["active_features"], dtype=bool)
    if not (np.all(np.isfinite(shift)) and np.all(np.isfinite(scale)) and np.all(scale > 0)):
        raise ValueError("Invalid normalizer parameters")
    valid = validity.astype(bool) & active[None]
    normalized = (values.astype(np.float64) - shift[None]) / scale[None]
    normalized[~valid] = 0.0
    x = np.concatenate([normalized.astype(np.float32), valid.astype(np.float32)], axis=1)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite neural inputs after normalization")
    return x



def make_model_input(tracking, manifest, normalizer):
    expected_schema = manifest["features"]
    if normalizer["feature_names"].tolist() != [feature["name"] for feature in expected_schema]:
        raise ValueError("Normalizer feature order differs from manifest")

    base, base_schema = make_features(
        tracking["ball_xy_m"], tracking["player_xy_m"], tracking["player_team"],
        tracking["frame_id"], tracking["period"], tracking["time_seconds"],
        tracking["pitch_length_m"], tracking["pitch_width_m"],
    )
    extra, extra_schema = build_extra(
        base, {"features": base_schema}, tracking, manifest["temporal_feature_config"],
    )
    if base.shape[1] != 78 or extra.shape[1] != 59 or base_schema + extra_schema != expected_schema:
        raise ValueError("Feature order or dimensions differ from the Native checkpoint")
    if not np.array_equal(base[:, 0].astype(bool), tracking["ball_available"]):
        raise ValueError("Unexpected ball availability feature")

    values = np.column_stack((base, extra))
    return apply_normalizer(values, np.isfinite(values), normalizer)
