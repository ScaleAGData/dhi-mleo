from typing import Any

import numpy as np
import xarray as xr
from tqdm import tqdm


def xr_predict(
    da: xr.DataArray,
    model: Any,
    feature_names,
    band_dim: str = "band",
    chunk_size: int | None = None,
    pred_name: str = "prediction",
) -> xr.DataArray:
    """
    Run model inference on a feature-stacked xarray.DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with shape [band_dim, y, x].
        The dimension specified by `band_dim` must correspond to model features.
    model : sklearn-like model
        Trained model implementing `.predict()`.
    feature_names : list of str
        Ordered list of feature names (must match model training order).
    band_dim : str, optional
        Name of the dimension representing feature bands (default: "band").
    chunk_size : int, optional
        Number of pixels per batch for memory-efficient prediction.
        If None, predict all at once.
    pred_name : str, optional
        Name for the output prediction DataArray.

    Returns
    -------
    da_pred : xr.DataArray
        Prediction DataArray with shape [y, x], same coords as input.
    """
    if band_dim not in da.dims:
        raise ValueError(
            f"Band dimension '{band_dim}' not found in DataArray dims {da.dims}."
        )
    if len(feature_names) != da.sizes[band_dim]:
        raise ValueError(
            f"Expected {len(feature_names)} features, got {da.sizes[band_dim]}."
        )

    da = da.transpose(band_dim, ...)

    n_bands = da.sizes[band_dim]
    n_pixels = np.prod([da.sizes[d] for d in da.dims if d != band_dim])
    data = da.values.reshape(n_bands, n_pixels).T  # [n_pixels, n_features]

    if chunk_size is None:
        y_pred = model.predict(data)
    else:
        preds = []
        for i in tqdm(range(0, n_pixels, chunk_size), desc="Predicting"):
            preds.append(model.predict(data[i : i + chunk_size]))
        y_pred = np.concatenate(preds, axis=0)

    spatial_dims = [d for d in da.dims if d != band_dim]
    da_pred = xr.DataArray(
        y_pred.reshape([da.sizes[d] for d in spatial_dims]),
        dims=spatial_dims,
        coords={d: da.coords[d] for d in spatial_dims},
        name=pred_name,
    )
    return da_pred
