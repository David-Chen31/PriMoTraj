import numpy as np


EARTH_RADIUS_M = 6371000.0


def denormalize_latlon(latlon_norm, mean, std):
    """Denormalize lat/lon using the first two dimensions of mean/std."""
    latlon_norm = np.asarray(latlon_norm)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    if mean.shape[0] < 2 or std.shape[0] < 2:
        raise ValueError('mean/std must contain at least 2 elements for lat/lon.')

    return latlon_norm * std[:2] + mean[:2]


def haversine_distance_meters(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in meters."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon * 0.5) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def ade_fde_from_normalized(pred_norm, true_norm, mean, std):
    """Compute ADE/FDE (meters) from normalized trajectory tensors.

    Args:
        pred_norm: shape (..., T, C) or (..., T, 2), C>=2.
        true_norm: same shape as pred_norm.
        mean/std: global normalization stats from npz, shape (9,) or at least (2,).

    Returns:
        ade_m: float, average displacement error in meters.
        fde_m: float, final displacement error in meters.
    """
    pred_norm = np.asarray(pred_norm)
    true_norm = np.asarray(true_norm)

    if pred_norm.shape != true_norm.shape:
        raise ValueError('pred_norm and true_norm must have the same shape.')
    if pred_norm.shape[-1] < 2:
        raise ValueError('Last dimension must be at least 2 (lat/lon).')

    pred_latlon = denormalize_latlon(pred_norm[..., :2], mean, std)
    true_latlon = denormalize_latlon(true_norm[..., :2], mean, std)

    dists = haversine_distance_meters(
        pred_latlon[..., 0],
        pred_latlon[..., 1],
        true_latlon[..., 0],
        true_latlon[..., 1],
    )

    # dists shape: (..., T)
    ade_m = float(np.mean(dists))
    fde_m = float(np.mean(dists[..., -1]))
    return ade_m, fde_m
