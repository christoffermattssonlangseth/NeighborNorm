"""Scanpy workflow to mitigate ubiquitous / territory genes in spatial data.

This module computes, per target gene:
1) Local neighborhood contrast scores.
2) Covariate-regressed residuals.
3) Optional process-field residuals (graph-smoothed subtraction).

It is robust to sparse AnnData matrices and does not overwrite ``adata.X``.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def _to_dense_vector(mat_col) -> np.ndarray:
    """Convert a 1D matrix/array-like column to a dense float vector."""
    if sparse.issparse(mat_col):
        arr = mat_col.toarray().ravel()
    else:
        arr = np.asarray(mat_col).ravel()
    return arr.astype(np.float64, copy=False)


def _get_gene_vector(adata: AnnData, gene: str, layer: str) -> np.ndarray:
    """Extract one gene vector from a layer as dense 1D float array."""
    gene_idx = adata.var_names.get_loc(gene)
    layer_data = adata.layers[layer]
    if sparse.issparse(layer_data):
        return _to_dense_vector(layer_data[:, gene_idx])
    return np.asarray(layer_data[:, gene_idx], dtype=np.float64).ravel()


def _compute_neighbor_mean(C, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute weighted neighbor mean using sparse/dense connectivities."""
    if sparse.issparse(C):
        numer = C.dot(x)
        denom = np.asarray(C.sum(axis=1)).ravel()
    else:
        numer = C @ x
        denom = np.asarray(C.sum(axis=1)).ravel()
    return np.asarray(numer).ravel() / (denom + eps)


def _extract_connectivities(adata: AnnData, key: str):
    """Version-agnostic retrieval of connectivities matrix for a neighbors key."""
    obsp_key = f"{key}_connectivities"
    if obsp_key in adata.obsp:
        return adata.obsp[obsp_key]

    # Scanpy stores key pointers in adata.uns[key].
    if key in adata.uns and isinstance(adata.uns[key], Mapping):
        uns_key = adata.uns[key]
        conn_key = uns_key.get("connectivities_key")
        if conn_key and conn_key in adata.obsp:
            return adata.obsp[conn_key]
        if "connectivities" in uns_key:
            return uns_key["connectivities"]

    # Conservative fallback: unique connectivities-like key containing the prefix.
    candidates = [k for k in adata.obsp.keys() if "connectivities" in k and key in k]
    if len(candidates) == 1:
        return adata.obsp[candidates[0]]

    raise KeyError(
        f"Could not find connectivities for key '{key}'. Expected '{obsp_key}' in "
        "adata.obsp or an explicit connectivities pointer in adata.uns."
    )


def _build_regression_design_matrix(
    adata: AnnData, local_density: np.ndarray, sample_key: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """Create OLS design matrix with available covariates and optional batch dummies."""
    n = adata.n_obs
    cols = []
    names = []

    # Intercept always included.
    cols.append(np.ones(n, dtype=np.float64))
    names.append("intercept")

    if "total_counts" in adata.obs:
        total_counts = np.asarray(adata.obs["total_counts"].values, dtype=np.float64)
        cols.append(np.log1p(np.clip(total_counts, a_min=0.0, a_max=None)))
        names.append("log1p_total_counts")

    if "area" in adata.obs:
        area = np.asarray(adata.obs["area"].values, dtype=np.float64)
        cols.append(np.log1p(np.clip(area, a_min=0.0, a_max=None)))
        names.append("log1p_area")

    cols.append(np.asarray(local_density, dtype=np.float64))
    names.append("local_density")

    if "batch" in adata.obs:
        batch_series = adata.obs["batch"].astype("category")
        dummies = pd.get_dummies(batch_series, prefix="batch", drop_first=True)
        for col in dummies.columns:
            cols.append(np.asarray(dummies[col].values, dtype=np.float64))
            names.append(str(col))

    # Optionally account for sample-specific offsets when sample IDs are available.
    if sample_key and sample_key in adata.obs and sample_key != "batch":
        sample_series = adata.obs[sample_key].astype("category")
        sample_dummies = pd.get_dummies(
            sample_series, prefix=str(sample_key), drop_first=True
        )
        for col in sample_dummies.columns:
            cols.append(np.asarray(sample_dummies[col].values, dtype=np.float64))
            names.append(str(col))

    X = np.column_stack(cols).astype(np.float64, copy=False)
    return X, names


def _fit_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals with finite-value masking and robust fallback."""
    resid = np.full(y.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < max(3, X.shape[1]):
        return resid

    Xv = X[valid]
    yv = y[valid]
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    yhat = Xv @ beta
    resid[valid] = yv - yhat
    return resid


def _safe_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Pearson correlation robust to constant vectors and non-finite values."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    xv = np.asarray(x[valid], dtype=np.float64)
    yv = np.asarray(y[valid], dtype=np.float64)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    if denom <= eps:
        return np.nan
    return float(np.sum(xv * yv) / denom)


def _moran_i(C, x: np.ndarray, eps: float = 1e-12) -> float:
    """Global Moran's I for one gene against the current connectivities matrix."""
    valid = np.isfinite(x)
    if valid.sum() < 3:
        return np.nan
    z = np.asarray(x[valid], dtype=np.float64)
    z = z - z.mean()
    denom = float(np.sum(z * z))
    if denom <= eps:
        return np.nan

    if sparse.issparse(C):
        C_use = C[valid][:, valid]
        s0 = float(C_use.sum())
        if s0 <= eps:
            return np.nan
        numer = float(z @ np.asarray(C_use.dot(z)).ravel())
    else:
        C_use = np.asarray(C)[np.ix_(valid, valid)]
        s0 = float(C_use.sum())
        if s0 <= eps:
            return np.nan
        numer = float(z @ (C_use @ z))

    n = z.shape[0]
    return float((n / s0) * (numer / (denom + eps)))


def _ensure_counts_layer(adata: AnnData, layer_counts: str = "counts") -> str:
    """Ensure raw counts are available in adata.layers[layer_counts]."""
    if layer_counts in adata.layers:
        return layer_counts

    # Copy from adata.X if explicit counts layer is absent.
    adata.layers[layer_counts] = adata.X.copy()
    print(f"[info] Created adata.layers['{layer_counts}'] from adata.X.")
    return layer_counts


def _prepare_log1p_cpm_layer(
    adata: AnnData, counts_layer: str, out_layer: str = "log1p_cpm"
) -> str:
    """Create log1p(CPM)-like layer without mutating adata.X or counts layer."""
    cpm_layer = "__tmp_cpm__"
    adata.layers[cpm_layer] = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(adata, target_sum=1e4, layer=cpm_layer, inplace=True)
    sc.pp.log1p(adata, layer=cpm_layer)
    adata.layers[out_layer] = adata.layers[cpm_layer].copy()
    del adata.layers[cpm_layer]
    print(f"[info] Created adata.layers['{out_layer}'] from '{counts_layer}'.")
    return out_layer


def _ensure_qc_covariates(adata: AnnData, counts_layer: str) -> None:
    """Ensure total_counts and n_genes_by_counts exist."""
    needed = {"total_counts", "n_genes_by_counts"}
    if needed.issubset(set(adata.obs.columns)):
        return
    sc.pp.calculate_qc_metrics(adata, layer=counts_layer, inplace=True)
    print("[info] Computed QC covariates: total_counts, n_genes_by_counts.")


def _ensure_spatial_neighbors(
    adata: AnnData,
    n_neighbors: int,
    key: str,
    sample_key: Optional[str] = None,
) -> None:
    """Build spatial-only neighbor graph, optionally constrained within samples."""
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required but missing.")
    if adata.obsm["spatial"].shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (n_cells, >=2).")

    if sample_key is None:
        adata.obsm["X_spatial"] = np.asarray(adata.obsm["spatial"])
        sc.pp.neighbors(
            adata, use_rep="X_spatial", n_neighbors=n_neighbors, key_added=key
        )
        print(
            f"[info] Built spatial neighbors with key='{key}', n_neighbors={n_neighbors}."
        )
        return

    if sample_key not in adata.obs:
        raise KeyError(
            f"sample_key='{sample_key}' not found in adata.obs columns: "
            f"{list(adata.obs.columns)}"
        )

    sample_values = (
        adata.obs[sample_key].astype("string").fillna("__missing_sample__").to_numpy()
    )
    unique_samples = pd.unique(sample_values)
    spatial = np.asarray(adata.obsm["spatial"])
    n_obs = adata.n_obs

    conn_rows: List[np.ndarray] = []
    conn_cols: List[np.ndarray] = []
    conn_vals: List[np.ndarray] = []
    dist_rows: List[np.ndarray] = []
    dist_cols: List[np.ndarray] = []
    dist_vals: List[np.ndarray] = []

    for sample in unique_samples:
        idx = np.where(sample_values == sample)[0]
        n_sub = idx.size
        if n_sub <= 1:
            continue
        k_sub = min(int(n_neighbors), n_sub - 1)
        sub = AnnData(X=np.zeros((n_sub, 1), dtype=np.float32))
        sub.obsm["X_spatial"] = spatial[idx]
        sc.pp.neighbors(sub, use_rep="X_spatial", n_neighbors=k_sub, key_added=key)

        conn_sub = _extract_connectivities(sub, key=key).tocoo()
        conn_rows.append(idx[conn_sub.row])
        conn_cols.append(idx[conn_sub.col])
        conn_vals.append(np.asarray(conn_sub.data, dtype=np.float64))

        dist_key = f"{key}_distances"
        if dist_key in sub.obsp:
            dist_sub = sub.obsp[dist_key].tocoo()
            dist_rows.append(idx[dist_sub.row])
            dist_cols.append(idx[dist_sub.col])
            dist_vals.append(np.asarray(dist_sub.data, dtype=np.float64))

        print(
            f"[info] Built sample-wise neighbors for {sample_key}='{sample}': "
            f"n_obs={n_sub}, n_neighbors={k_sub}."
        )

    if conn_rows:
        C = sparse.csr_matrix(
            (
                np.concatenate(conn_vals),
                (np.concatenate(conn_rows), np.concatenate(conn_cols)),
            ),
            shape=(n_obs, n_obs),
        )
    else:
        C = sparse.csr_matrix((n_obs, n_obs), dtype=np.float64)
    adata.obsp[f"{key}_connectivities"] = C

    if dist_rows:
        D = sparse.csr_matrix(
            (
                np.concatenate(dist_vals),
                (np.concatenate(dist_rows), np.concatenate(dist_cols)),
            ),
            shape=(n_obs, n_obs),
        )
        adata.obsp[f"{key}_distances"] = D

    adata.uns[key] = {
        "connectivities_key": f"{key}_connectivities",
        "distances_key": f"{key}_distances",
        "params": {
            "n_neighbors": int(n_neighbors),
            "use_rep": "spatial",
            "sample_key": sample_key,
            "sample_aware": True,
        },
    }
    print(
        f"[info] Built sample-aware spatial neighbors with key='{key}', "
        f"n_neighbors={n_neighbors}, sample_key='{sample_key}', n_samples={len(unique_samples)}."
    )


def _resolve_stickiness_score_weights(
    score_weights: Optional[Mapping[str, float]] = None,
) -> Mapping[str, float]:
    """Validate and normalize ranking weights."""
    default = {
        "neighbor_corr": 0.45,
        "moran_i": 0.45,
        "contrast_mad": 0.10,
        "detect_frac": 0.0,
    }
    if score_weights is None:
        return default

    unknown = set(score_weights.keys()) - set(default.keys())
    if unknown:
        raise KeyError(
            f"Unknown score_weights keys: {sorted(unknown)}. "
            f"Allowed keys: {sorted(default.keys())}."
        )

    merged = default.copy()
    merged.update({k: float(v) for k, v in score_weights.items()})
    if any(v < 0 for v in merged.values()):
        raise ValueError("All score weights must be >= 0.")

    total = float(sum(merged.values()))
    if total <= 0:
        raise ValueError("At least one score weight must be > 0.")
    return {k: v / total for k, v in merged.items()}


def _compute_stickiness_metrics(
    counts_data,
    log_data,
    C,
    var_names: Sequence[str],
    min_detect_frac: float = 0.05,
    min_mean_log1p: float = 0.1,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute per-gene stickiness component metrics for one subset."""
    n_obs = counts_data.shape[0]
    n_vars = counts_data.shape[1]

    if sparse.issparse(counts_data):
        detect_frac = np.asarray((counts_data > 0).sum(axis=0)).ravel() / max(1, n_obs)
    else:
        detect_frac = np.mean(np.asarray(counts_data) > 0, axis=0)
    detect_frac = np.asarray(detect_frac, dtype=np.float64).ravel()

    if sparse.issparse(log_data):
        mean_log1p = np.asarray(log_data.mean(axis=0)).ravel().astype(np.float64)
        log_data_view = log_data.tocsc()
    else:
        mean_log1p = np.asarray(log_data, dtype=np.float64).mean(axis=0).ravel()
        log_data_view = np.asarray(log_data, dtype=np.float64)

    eligible = (detect_frac >= min_detect_frac) & (mean_log1p >= min_mean_log1p)
    neighbor_corr = np.full(n_vars, np.nan, dtype=np.float64)
    moran_i = np.full(n_vars, np.nan, dtype=np.float64)
    contrast_mad = np.full(n_vars, np.nan, dtype=np.float64)

    idxs = np.where(eligible)[0]
    for j in idxs:
        if sparse.issparse(log_data_view):
            x = _to_dense_vector(log_data_view[:, j])
        else:
            x = np.asarray(log_data_view[:, j], dtype=np.float64).ravel()

        nbr_mean = _compute_neighbor_mean(C, x=x, eps=eps)
        contrast = x - nbr_mean
        centered = contrast - np.nanmedian(contrast)

        neighbor_corr[j] = _safe_pearson(x, nbr_mean, eps=eps)
        moran_i[j] = _moran_i(C, x, eps=eps)
        contrast_mad[j] = np.nanmedian(np.abs(centered))

    return pd.DataFrame(
        {
            "detect_frac": detect_frac,
            "mean_log1p": mean_log1p,
            "eligible": eligible,
            "neighbor_corr": neighbor_corr,
            "moran_i": moran_i,
            "contrast_mad": contrast_mad,
        },
        index=pd.Index(var_names, dtype="string"),
    )


def _rank_spatial_stickiness(
    adata: AnnData,
    C,
    counts_layer: str,
    log_layer: str,
    sample_key: Optional[str] = None,
    min_detect_frac: float = 0.05,
    min_mean_log1p: float = 0.1,
    min_samples_eligible_frac: float = 0.5,
    score_weights: Optional[Mapping[str, float]] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Rank genes by spatial stickiness and return a per-gene metrics table."""
    if not (0.0 <= min_detect_frac <= 1.0):
        raise ValueError("min_detect_frac must be in [0, 1].")
    if not (0.0 <= min_samples_eligible_frac <= 1.0):
        raise ValueError("min_samples_eligible_frac must be in [0, 1].")

    weights = _resolve_stickiness_score_weights(score_weights=score_weights)
    counts_data = adata.layers[counts_layer]
    log_data = adata.layers[log_layer]
    var_names = adata.var_names.astype(str)

    ranking: pd.DataFrame
    if sample_key is not None and sample_key in adata.obs:
        sample_values = (
            adata.obs[sample_key]
            .astype("string")
            .fillna("__missing_sample__")
            .to_numpy()
        )
        unique_samples = pd.unique(sample_values)
        metrics_by_sample = []

        for sample in unique_samples:
            idx = np.where(sample_values == sample)[0]
            if idx.size < 2:
                continue

            if sparse.issparse(C):
                C_sub = C[idx][:, idx]
            else:
                C_sub = np.asarray(C)[np.ix_(idx, idx)]

            metrics_sub = _compute_stickiness_metrics(
                counts_data=counts_data[idx],
                log_data=log_data[idx],
                C=C_sub,
                var_names=var_names,
                min_detect_frac=min_detect_frac,
                min_mean_log1p=min_mean_log1p,
                eps=eps,
            ).reset_index(names="gene")
            metrics_sub["sample"] = str(sample)
            metrics_by_sample.append(metrics_sub)

        if metrics_by_sample:
            long_df = pd.concat(metrics_by_sample, axis=0, ignore_index=True)
            n_samples_total = len(metrics_by_sample)
            min_samples_required = max(
                1, int(np.ceil(float(min_samples_eligible_frac) * n_samples_total))
            )

            grouped = long_df.groupby("gene", sort=False)
            ranking = grouped.agg(
                detect_frac=("detect_frac", "median"),
                mean_log1p=("mean_log1p", "median"),
                neighbor_corr=("neighbor_corr", "median"),
                moran_i=("moran_i", "median"),
                contrast_mad=("contrast_mad", "median"),
                n_samples_eligible=("eligible", lambda s: int(np.sum(s.astype(bool)))),
            )
            ranking["n_samples_total"] = int(n_samples_total)
            ranking["eligible"] = (
                ranking["n_samples_eligible"].fillna(0).astype(int) >= min_samples_required
            )
            ranking = (
                pd.DataFrame(index=pd.Index(var_names, dtype="string"))
                .join(ranking, how="left")
                .fillna({"n_samples_eligible": 0, "n_samples_total": n_samples_total})
            )
            ranking["n_samples_eligible"] = ranking["n_samples_eligible"].astype(int)
            ranking["n_samples_total"] = ranking["n_samples_total"].astype(int)
        else:
            ranking = pd.DataFrame(index=pd.Index(var_names, dtype="string"))
            ranking["detect_frac"] = np.nan
            ranking["mean_log1p"] = np.nan
            ranking["neighbor_corr"] = np.nan
            ranking["moran_i"] = np.nan
            ranking["contrast_mad"] = np.nan
            ranking["n_samples_eligible"] = 0
            ranking["n_samples_total"] = 0
            ranking["eligible"] = False
    else:
        ranking = _compute_stickiness_metrics(
            counts_data=counts_data,
            log_data=log_data,
            C=C,
            var_names=var_names,
            min_detect_frac=min_detect_frac,
            min_mean_log1p=min_mean_log1p,
            eps=eps,
        )
        ranking["n_samples_eligible"] = ranking["eligible"].astype(int)
        ranking["n_samples_total"] = 1

    rank_corr = ranking["neighbor_corr"].rank(pct=True, na_option="bottom")
    rank_moran = ranking["moran_i"].rank(pct=True, na_option="bottom")
    rank_detect = ranking["detect_frac"].rank(pct=True, na_option="bottom")
    rank_mad = ranking["contrast_mad"].rank(
        pct=True, ascending=False, na_option="bottom"
    )
    ranking["stickiness_score"] = (
        weights["neighbor_corr"] * rank_corr
        + weights["moran_i"] * rank_moran
        + weights["detect_frac"] * rank_detect
        + weights["contrast_mad"] * rank_mad
    )
    ranking.loc[~ranking["eligible"].astype(bool), "stickiness_score"] = np.nan
    ranking["score_weights"] = str(dict(weights))
    ranking = ranking.sort_values(
        by=["stickiness_score", "moran_i", "neighbor_corr"], ascending=False
    )
    return ranking

def discover_spatially_sticky_genes(
    adata: AnnData,
    n_neighbors: int = 15,
    key: str = "spatial_neighbors",
    sample_key: Optional[str] = None,
    layer_counts: str = "counts",
    min_detect_frac: float = 0.05,
    min_mean_log1p: float = 0.1,
    min_samples_eligible_frac: float = 0.5,
    score_weights: Optional[Mapping[str, float]] = None,
    top_k: int = 20,
    ranking_key: str = "sticky_gene_ranking",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Discover and rank sticky genes across all genes.

    Returns a ranked DataFrame and stores results in:
    - ``adata.uns[ranking_key]`` (full table)
    - ``adata.uns[f"{ranking_key}_top_genes"]`` (top gene symbols)

    If ``sample_key`` is provided, neighbors are built within each sample only.
    The ranking metrics are then computed per sample and aggregated by median.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    counts_layer = _ensure_counts_layer(adata, layer_counts=layer_counts)
    _ensure_qc_covariates(adata, counts_layer=counts_layer)
    log_layer = _prepare_log1p_cpm_layer(
        adata, counts_layer=counts_layer, out_layer="log1p_cpm"
    )
    _ensure_spatial_neighbors(
        adata, n_neighbors=n_neighbors, key=key, sample_key=sample_key
    )
    C = _extract_connectivities(adata, key=key)

    ranking = _rank_spatial_stickiness(
        adata=adata,
        C=C,
        counts_layer=counts_layer,
        log_layer=log_layer,
        sample_key=sample_key,
        min_detect_frac=min_detect_frac,
        min_mean_log1p=min_mean_log1p,
        min_samples_eligible_frac=min_samples_eligible_frac,
        score_weights=score_weights,
        eps=eps,
    )
    top_genes = (
        ranking.index[ranking["stickiness_score"].notna()].to_list()[: int(top_k)]
    )
    adata.uns[ranking_key] = ranking.copy()
    adata.uns[f"{ranking_key}_top_genes"] = top_genes
    print(
        f"[info] Ranked sticky genes (eligible={int(ranking['eligible'].sum())}, "
        f"top_k={top_k}). Stored in adata.uns['{ranking_key}']."
    )
    return ranking


def compute_spatial_sticky_gene_corrections(
    adata: AnnData,
    sticky_genes: Optional[Sequence[str]] = None,
    n_neighbors: int = 15,
    key: str = "spatial_neighbors",
    sample_key: Optional[str] = None,
    layer_counts: str = "counts",
    do_field_residual: bool = False,
    discover_additional_sticky: bool = False,
    additional_top_k: int = 10,
    min_detect_frac: float = 0.05,
    min_mean_log1p: float = 0.1,
    min_samples_eligible_frac: float = 0.5,
    score_weights: Optional[Mapping[str, float]] = None,
    sticky_ranking_key: str = "sticky_gene_ranking",
    eps: float = 1e-12,
) -> AnnData:
    """Compute spatial sticky-gene correction features and store in adata.obs.

    Parameters
    ----------
    adata
        Input AnnData. Raw counts in ``adata.X`` or ``adata.layers[layer_counts]``.
    sticky_genes
        List of target ubiquitous genes (e.g., ["MBP", "GFAP"]).
    discover_additional_sticky
        If True, rank all genes by stickiness and include top genes not already listed.
    additional_top_k
        Number of extra discovered sticky genes to include when discovery is enabled.
    min_detect_frac
        Minimum fraction of spots with non-zero counts for a gene to be eligible.
    min_mean_log1p
        Minimum mean log1p CPM expression for a gene to be eligible.
    min_samples_eligible_frac
        Fraction of samples where a gene must be eligible when ``sample_key`` is used.
    score_weights
        Optional weights for ranking components. Keys:
        ``neighbor_corr``, ``moran_i``, ``contrast_mad``, ``detect_frac``.
    sticky_ranking_key
        ``adata.uns`` key where sticky-gene ranking table is stored.
    n_neighbors
        Number of neighbors for spatial graph construction.
    key
        Neighbors key for Scanpy graph objects.
    sample_key
        Optional column in ``adata.obs`` (e.g., ``sample_id``) used to build
        within-sample neighbor graphs and sample dummies in regression.
    layer_counts
        Name of counts layer to use/create.
    do_field_residual
        If True, also compute ``gene_field_resid = x - smooth_x``.
    eps
        Small value for numerical stability.

    Returns
    -------
    AnnData
        Same object with added columns in ``adata.obs`` and layers.
    """
    assert isinstance(adata, AnnData), "adata must be an AnnData object."

    counts_layer = _ensure_counts_layer(adata, layer_counts=layer_counts)
    _ensure_qc_covariates(adata, counts_layer=counts_layer)
    log_layer = _prepare_log1p_cpm_layer(
        adata, counts_layer=counts_layer, out_layer="log1p_cpm"
    )

    _ensure_spatial_neighbors(
        adata, n_neighbors=n_neighbors, key=key, sample_key=sample_key
    )
    C = _extract_connectivities(adata, key=key)
    if C.shape[0] != adata.n_obs or C.shape[1] != adata.n_obs:
        raise ValueError(
            f"Connectivities matrix has shape {C.shape}, expected ({adata.n_obs}, {adata.n_obs})."
        )

    local_density = np.asarray(C.sum(axis=1)).ravel().astype(np.float64)
    adata.obs["local_density"] = local_density
    design_X, design_names = _build_regression_design_matrix(
        adata, local_density=local_density, sample_key=sample_key
    )
    print(f"[info] Regression covariates: {', '.join(design_names)}")

    requested_genes = list(sticky_genes) if sticky_genes is not None else []
    if not requested_genes and not discover_additional_sticky:
        raise ValueError(
            "Provide sticky_genes or set discover_additional_sticky=True to auto-detect."
        )

    discovered_genes: List[str] = []
    if discover_additional_sticky:
        if additional_top_k <= 0:
            raise ValueError("additional_top_k must be > 0 when discovery is enabled.")
        ranking = _rank_spatial_stickiness(
            adata=adata,
            C=C,
            counts_layer=counts_layer,
            log_layer=log_layer,
            sample_key=sample_key,
            min_detect_frac=min_detect_frac,
            min_mean_log1p=min_mean_log1p,
            min_samples_eligible_frac=min_samples_eligible_frac,
            score_weights=score_weights,
            eps=eps,
        )
        adata.uns[sticky_ranking_key] = ranking.copy()
        ranked = ranking.index[ranking["stickiness_score"].notna()].to_list()
        requested_set = set(requested_genes)
        discovered_genes = [g for g in ranked if g not in requested_set][:additional_top_k]
        adata.uns[f"{sticky_ranking_key}_top_genes"] = ranked[:additional_top_k]
        if discovered_genes:
            print(
                "[info] Added discovered sticky genes: "
                + ", ".join(discovered_genes[: min(10, len(discovered_genes))])
                + (" ..." if len(discovered_genes) > 10 else "")
            )
        else:
            print("[info] Discovery ran, but no additional eligible sticky genes were found.")

    final_requested = list(dict.fromkeys(requested_genes + discovered_genes))
    present_genes = [g for g in final_requested if g in adata.var_names]
    missing_genes = [g for g in final_requested if g not in adata.var_names]
    if missing_genes:
        print(f"[warn] Skipping absent genes: {missing_genes}")
    if not present_genes:
        print("[warn] No sticky genes present in adata.var_names; nothing to compute.")
        return adata

    for g in present_genes:
        x = _get_gene_vector(adata, gene=g, layer=log_layer)
        nbr_mean = _compute_neighbor_mean(C, x=x, eps=eps)

        # x and nbr_mean are on the same log1p CPM scale, so subtraction is coherent.
        contrast = x - nbr_mean
        adata.obs[f"{g}_local_contrast"] = contrast

        resid = _fit_residuals(y=x, X=design_X)
        adata.obs[f"{g}_resid"] = resid

        if do_field_residual:
            smooth_x = nbr_mean
            adata.obs[f"{g}_field_resid"] = x - smooth_x

        print(
            f"[info] Computed features for {g}: local_contrast, resid"
            f"{', field_resid' if do_field_residual else ''}."
        )

    return adata


def _build_tiled_spatial_basis(
    adata: AnnData,
    sample_key: str,
    out_basis: str = "spatial_tiled",
    gap_fraction: float = 0.15,
) -> str:
    """Create a non-overlapping spatial basis by offsetting samples along x-axis."""
    if sample_key not in adata.obs:
        raise KeyError(f"sample_key='{sample_key}' not found in adata.obs.")
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required for tiled spatial plotting.")

    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (n_cells, >=2).")

    coords = spatial[:, :2].copy()
    sample_values = adata.obs[sample_key].astype("string").fillna("__missing_sample__")
    categories = pd.Index(pd.unique(sample_values.to_numpy()))
    x_min = float(np.nanmin(coords[:, 0]))
    x_max = float(np.nanmax(coords[:, 0]))
    span_x = max(x_max - x_min, 1.0)
    gap = span_x * float(gap_fraction)

    out = coords.copy()
    for i, sample in enumerate(categories):
        mask = sample_values.to_numpy() == sample
        out[mask, 0] = coords[mask, 0] + i * (span_x + gap)

    adata.obsm[out_basis] = out
    return out_basis


def plot_sticky_gene_views(
    adata: AnnData,
    genes: Sequence[str],
    spot_size: float = 30.0,
    cmap: str = "viridis",
    sample_key: Optional[str] = None,
    tile_samples: bool = True,
    tile_basis: str = "spatial_tiled",
    tile_gap_fraction: float = 0.15,
) -> None:
    """Visualization helper for spatial and UMAP views of correction outputs.

    When sample IDs share similar coordinate ranges, set ``sample_key`` to tile
    samples in one non-overlapping spatial basis before plotting.
    """
    spatial_basis = "spatial"
    if (
        tile_samples
        and sample_key is not None
        and sample_key in adata.obs
        and adata.obs[sample_key].nunique(dropna=False) > 1
    ):
        spatial_basis = _build_tiled_spatial_basis(
            adata,
            sample_key=sample_key,
            out_basis=tile_basis,
            gap_fraction=tile_gap_fraction,
        )
        print(
            f"[info] Using tiled spatial basis '{spatial_basis}' with sample_key='{sample_key}'."
        )

    for g in genes:
        cols = [
            c
            for c in [g, f"{g}_local_contrast", f"{g}_resid"]
            if c in adata.obs.columns or c in adata.var_names
        ]
        if not cols:
            print(f"[warn] No plottable columns found for gene '{g}'.")
            continue

        if spatial_basis == "spatial":
            # Prefer sc.pl.spatial when possible; fallback to embedding if image metadata is absent.
            try:
                sc.pl.spatial(adata, color=cols, spot_size=spot_size, cmap=cmap)
            except Exception as e:
                print(
                    f"[warn] sc.pl.spatial failed for '{g}' ({e}); using sc.pl.embedding(basis='spatial')."
                )
                sc.pl.embedding(
                    adata, basis="spatial", color=cols, size=spot_size, cmap=cmap
                )
        else:
            sc.pl.embedding(
                adata,
                basis=spatial_basis,
                color=cols,
                size=spot_size,
                cmap=cmap,
            )

        if "X_umap" in adata.obsm:
            sc.pl.umap(adata, color=cols, cmap=cmap)


if __name__ == "__main__":
    # Minimal usage example (assumes `adata` already loaded).
    #
    # import scanpy as sc
    # adata = sc.read_h5ad("your_data.h5ad")
    # sticky_genes = ["MBP", "GFAP"]
    # # Optional discovery step:
    # # ranking = discover_spatially_sticky_genes(adata, top_k=20)
    # # print(ranking.head(20))
    # adata = compute_spatial_sticky_gene_corrections(
    #     adata,
    #     sticky_genes=sticky_genes,
    #     n_neighbors=15,
    #     key="spatial_neighbors",
    #     sample_key="sample_id",
    #     layer_counts="counts",
    #     do_field_residual=True,
    #     discover_additional_sticky=True,
    #     additional_top_k=10,
    # )
    # plot_sticky_gene_views(adata, sticky_genes)
    pass
