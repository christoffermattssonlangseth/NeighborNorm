"""Cell-type-adjusted spatial stickiness scoring for AnnData.

This module addresses a common failure mode in spatial "stickiness" ranking:
genes can appear spatially smooth simply because cell types are spatially
clustered. We correct for that by residualizing expression against cell type
and library size, then scoring smoothness on residuals.

Pipeline:
1) Residualize expression:
      y_g ~ 1 + cell_type + log_total_counts
2) Compute graph-Laplacian stickiness on residuals:
      stickiness = 1 - (r^T L r) / (r^T D r + eps)
3) Conditional permutation null:
   shuffle residuals within cell type (optionally within UMI bins) and
   compute z-score, p-value, q-value (BH-FDR).
4) Optional within-cell-type stickiness:
   compute per-cell-type stickiness and aggregate (weighted mean + max).

Only numpy/scipy/pandas are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class ResidualizationResult:
    """Container for residualization outputs."""

    residuals: np.ndarray
    normalized_log_expr: object
    design_matrix: np.ndarray
    design_columns: List[str]
    log_total_counts: np.ndarray
    cell_type: np.ndarray


def _get_matrix(adata, layer: Optional[str] = None):
    """Return expression matrix from `adata.X` or `adata.layers[layer]`."""
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")
    return adata.layers[layer]


def _row_sums(mat) -> np.ndarray:
    """Fast row sums for dense/sparse matrices."""
    if sparse.issparse(mat):
        return np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(mat, dtype=np.float64).sum(axis=1).ravel()


def _compute_total_counts(
    adata,
    mat,
    total_counts_key: str = "total_counts",
) -> np.ndarray:
    """Get per-cell total counts from obs, or compute from matrix if missing."""
    if total_counts_key in adata.obs:
        total_counts = np.asarray(adata.obs[total_counts_key].values, dtype=np.float64)
    else:
        total_counts = _row_sums(mat)
        adata.obs[total_counts_key] = total_counts
    # Guard against zero / negative totals.
    return np.clip(total_counts, a_min=1.0, a_max=None)


def _normalize_log1p_counts(
    mat,
    total_counts: np.ndarray,
    scale_factor: float = 1e4,
) -> np.ndarray:
    """Compute log1p(scale_factor * counts / total_counts) matrix."""
    scale = (scale_factor / total_counts).astype(np.float64)
    if sparse.issparse(mat):
        mat_use = mat.tocsr(copy=True).astype(np.float64)
        mat_use = mat_use.multiply(scale[:, None])
        mat_use.data = np.log1p(mat_use.data)
        return mat_use
    dense = np.asarray(mat, dtype=np.float64)
    return np.log1p(dense * scale[:, None])


def _build_design_matrix(
    cell_type: np.ndarray,
    log_total_counts: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Build design matrix: intercept + one-hot(cell_type, drop_first) + log_total_counts."""
    n = cell_type.shape[0]
    ct_cat = pd.Categorical(cell_type)
    ct_dummies = pd.get_dummies(ct_cat, prefix="cell_type", drop_first=True)

    cols = [np.ones(n, dtype=np.float64)]
    names = ["intercept"]
    if ct_dummies.shape[1] > 0:
        cols.append(np.asarray(ct_dummies.values, dtype=np.float64))
        names.extend([str(c) for c in ct_dummies.columns])
    cols.append(np.asarray(log_total_counts, dtype=np.float64).reshape(-1, 1))
    names.append("log_total_counts")

    X = np.column_stack(cols).astype(np.float64, copy=False)
    return X, names


def _to_dense_gene_block(mat, start: int, end: int) -> np.ndarray:
    """Extract gene block [start:end] as dense float64."""
    if sparse.issparse(mat):
        return mat[:, start:end].toarray().astype(np.float64, copy=False)
    return np.asarray(mat[:, start:end], dtype=np.float64)


def compute_celltype_residuals(
    adata,
    layer: Optional[str] = None,
    cell_type_key: str = "cell_type",
    total_counts_key: str = "total_counts",
    out_layer: Optional[str] = "sticky_resid",
    scale_factor: float = 1e4,
    chunk_size: int = 512,
    dtype=np.float32,
) -> ResidualizationResult:
    """Residualize gene expression for cell type and library size.

    Model per gene g:
        y_g ~ 1 + cell_type + log_total_counts

    where:
        y = log1p(scale_factor * counts / total_counts)

    Parameters
    ----------
    adata
        AnnData object.
    layer
        Layer to use as input counts. If None, uses `adata.X`.
    cell_type_key
        `adata.obs` column with cell-type labels.
    total_counts_key
        `adata.obs` column with per-cell total counts; computed from input matrix if missing.
    out_layer
        If not None, stores residual matrix into `adata.layers[out_layer]`.
    scale_factor
        Scale factor for count normalization before log1p.
    chunk_size
        Number of genes to process per least-squares block.
    dtype
        Output dtype for residual matrix.

    Returns
    -------
    ResidualizationResult
        Residual matrix and supporting intermediate values.
    """
    if cell_type_key not in adata.obs:
        raise KeyError(f"'{cell_type_key}' not found in adata.obs.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    mat = _get_matrix(adata, layer=layer)
    total_counts = _compute_total_counts(adata, mat, total_counts_key=total_counts_key)
    log_total = np.log1p(total_counts)
    cell_type = (
        adata.obs[cell_type_key].astype("string").fillna("__missing_cell_type__").to_numpy()
    )

    y = _normalize_log1p_counts(mat, total_counts=total_counts, scale_factor=scale_factor)
    if sparse.issparse(y):
        y = y.tocsc()

    X, design_cols = _build_design_matrix(cell_type=cell_type, log_total_counts=log_total)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    resid = np.empty((n_cells, n_genes), dtype=dtype)

    # Solve all genes in chunks: beta = argmin ||X beta - Y||_2
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        y_block = _to_dense_gene_block(y, start, end)
        beta, *_ = np.linalg.lstsq(X, y_block, rcond=None)
        fit_block = X @ beta
        resid[:, start:end] = (y_block - fit_block).astype(dtype, copy=False)

    if out_layer is not None:
        adata.layers[out_layer] = resid

    return ResidualizationResult(
        residuals=resid,
        normalized_log_expr=y,
        design_matrix=X,
        design_columns=design_cols,
        log_total_counts=log_total,
        cell_type=cell_type,
    )


def _residualize_by_cell_type(
    adata,
    layer: Optional[str] = None,
    cell_type_key: str = "cell_type",
    total_counts_key: str = "total_counts",
    key_added: str = "stickiness",
    store_residuals: bool = False,
    scale_factor: float = 1e4,
    chunk_size: int = 512,
    dtype=np.float32,
) -> ResidualizationResult:
    """Residualize expression by cell type and log library size.

    Residualization model (per gene):
        y_g ~ 1 + cell_type + log_total_counts

    where:
        y = log1p(scale_factor * counts / total_counts)
    """
    out_layer = f"{key_added}_resid" if store_residuals else None
    return compute_celltype_residuals(
        adata=adata,
        layer=layer,
        cell_type_key=cell_type_key,
        total_counts_key=total_counts_key,
        out_layer=out_layer,
        scale_factor=scale_factor,
        chunk_size=chunk_size,
        dtype=dtype,
    )


def _get_connectivities(adata, connectivities_key: str = "connectivities"):
    """Get connectivities matrix from adata.obsp and return CSR sparse matrix."""
    if connectivities_key not in adata.obsp:
        raise KeyError(f"'{connectivities_key}' not found in adata.obsp.")
    W = adata.obsp[connectivities_key]
    if sparse.issparse(W):
        return W.tocsr()
    return sparse.csr_matrix(np.asarray(W, dtype=np.float64))


def _laplacian_stickiness(
    residuals,
    W: sparse.csr_matrix,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute residual-based Laplacian roughness and normalized stickiness.

    rough_g = r_g^T (D - W) r_g
    denom_g = r_g^T D r_g + eps
    stickiness_g = 1 - rough_g / denom_g
    """
    d = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    if sparse.issparse(residuals):
        R = residuals.tocsr().astype(np.float64)
        WR = W.dot(R)
        dr2 = np.asarray(R.power(2).T.dot(d)).ravel()
        rwr = np.asarray(R.multiply(WR).sum(axis=0)).ravel()
    else:
        R = np.asarray(residuals, dtype=np.float64)
        WR = W.dot(R)
        dr2 = np.sum((R * R) * d[:, None], axis=0)
        rwr = np.sum(R * WR, axis=0)
    rough = dr2 - rwr
    denom = dr2 + eps
    score = 1.0 - (rough / denom)
    score = np.where(dr2 > eps, score, np.nan)
    return score, rough, denom


def compute_stickiness_scores(
    adata,
    residuals: np.ndarray,
    connectivities_key: str = "connectivities",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute residual-based stickiness scores for all genes."""
    W = _get_connectivities(adata, connectivities_key=connectivities_key)
    score, rough, denom = _laplacian_stickiness(residuals=residuals, W=W, eps=eps)
    return pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "stickiness_raw": score,
            "roughness": rough,
            "denom": denom,
        }
    )


def _build_permutation_groups(
    cell_type: np.ndarray,
    log_total_counts: np.ndarray,
    umi_n_bins: Optional[int] = None,
) -> List[np.ndarray]:
    """Build index groups for conditional shuffling."""
    groups: List[np.ndarray] = []
    ct = pd.Series(cell_type, dtype="string").fillna("__missing_cell_type__").to_numpy()
    unique_ct = pd.unique(ct)

    for label in unique_ct:
        idx = np.where(ct == label)[0]
        if idx.size < 2:
            continue
        if umi_n_bins is None or umi_n_bins <= 1:
            groups.append(idx)
            continue

        values = log_total_counts[idx]
        q = np.linspace(0.0, 1.0, int(umi_n_bins) + 1)
        edges = np.quantile(values, q)
        edges = np.unique(edges)
        if edges.size <= 2:
            groups.append(idx)
            continue
        bins = np.digitize(values, edges[1:-1], right=True)
        for b in np.unique(bins):
            bidx = idx[bins == b]
            if bidx.size >= 2:
                groups.append(bidx)
    return groups


def _permute_within_groups(
    n_cells: int,
    groups: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return permutation index that shuffles entries within each group."""
    perm = np.arange(n_cells)
    for g in groups:
        perm[g] = rng.permutation(g)
    return perm


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (supports NaNs)."""
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m, dtype=np.float64) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    out = np.empty_like(ranked)
    out[order] = adj
    q[valid] = out
    return q


def permutation_null_stickiness(
    residuals: np.ndarray,
    observed_scores: np.ndarray,
    W: sparse.csr_matrix,
    cell_type: np.ndarray,
    log_total_counts: np.ndarray,
    n_perm: int = 200,
    random_state: Optional[int] = 0,
    umi_n_bins: Optional[int] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Conditional permutation null for residual-based stickiness.

    Residuals are shuffled within cell type (and optional UMI bins) per permutation.
    """
    if n_perm <= 0:
        raise ValueError("n_perm must be > 0.")
    rng = np.random.default_rng(random_state)
    n_cells, n_genes = residuals.shape

    groups = _build_permutation_groups(
        cell_type=cell_type,
        log_total_counts=log_total_counts,
        umi_n_bins=umi_n_bins,
    )
    if len(groups) == 0:
        raise ValueError("No valid permutation groups. Check cell_type and umi_n_bins.")

    null_sum = np.zeros(n_genes, dtype=np.float64)
    null_sumsq = np.zeros(n_genes, dtype=np.float64)
    null_n = np.zeros(n_genes, dtype=np.int64)
    ge_count = np.zeros(n_genes, dtype=np.int64)
    obs = np.asarray(observed_scores, dtype=np.float64)

    for _ in range(int(n_perm)):
        perm_idx = _permute_within_groups(n_cells=n_cells, groups=groups, rng=rng)
        perm_scores, _, _ = _laplacian_stickiness(residuals[perm_idx, :], W=W, eps=eps)

        finite = np.isfinite(perm_scores)
        null_sum += np.where(finite, perm_scores, 0.0)
        null_sumsq += np.where(finite, perm_scores**2, 0.0)
        null_n += finite.astype(np.int64)
        ge_count += np.where(np.isfinite(obs), perm_scores >= obs, False)

    null_mean = np.divide(
        null_sum,
        np.maximum(null_n, 1),
        out=np.full(n_genes, np.nan, dtype=np.float64),
        where=null_n > 0,
    )
    null_var = np.maximum(
        np.divide(
            null_sumsq,
            np.maximum(null_n, 1),
            out=np.full(n_genes, np.nan, dtype=np.float64),
            where=null_n > 0,
        )
        - null_mean**2,
        0.0,
    )
    null_sd = np.sqrt(null_var)
    z = (obs - null_mean) / (null_sd + eps)
    pval = (1.0 + ge_count) / (float(n_perm) + 1.0)

    # Keep NaN status for genes with non-finite observed scores.
    bad = ~np.isfinite(obs) | (null_n == 0)
    null_mean[bad] = np.nan
    null_sd[bad] = np.nan
    z[bad] = np.nan
    pval[bad] = np.nan
    qval = _bh_fdr(pval)

    return pd.DataFrame(
        {
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z": z,
            "pval": pval,
            "qval": qval,
        }
    )


def within_celltype_stickiness(
    residuals: np.ndarray,
    W: sparse.csr_matrix,
    cell_type: np.ndarray,
    min_cells: int = 30,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute stickiness within each cell type and aggregate across types."""
    ct = pd.Series(cell_type, dtype="string").fillna("__missing_cell_type__").to_numpy()
    labels = pd.unique(ct)
    per_ct_scores: List[np.ndarray] = []
    per_ct_n: List[int] = []

    for label in labels:
        idx = np.where(ct == label)[0]
        if idx.size < int(min_cells):
            continue
        W_sub = W[idx][:, idx]
        R_sub = residuals[idx, :]
        s_sub, _, _ = _laplacian_stickiness(R_sub, W_sub, eps=eps)
        per_ct_scores.append(s_sub)
        per_ct_n.append(idx.size)

    n_genes = residuals.shape[1]
    if len(per_ct_scores) == 0:
        return pd.DataFrame(
            {
                "stickiness_withinCT_mean": np.full(n_genes, np.nan),
                "stickiness_withinCT_max": np.full(n_genes, np.nan),
            }
        )

    S = np.vstack(per_ct_scores)  # (n_ct, n_genes)
    w = np.asarray(per_ct_n, dtype=np.float64)[:, None]
    valid = np.isfinite(S)
    weighted_sum = np.sum(np.where(valid, S, 0.0) * w, axis=0)
    weight_tot = np.sum(np.where(valid, w, 0.0), axis=0)
    mean_score = np.divide(
        weighted_sum,
        weight_tot,
        out=np.full(n_genes, np.nan, dtype=np.float64),
        where=weight_tot > 0,
    )
    all_nan = np.all(~np.isfinite(S), axis=0)
    max_score = np.max(np.where(np.isfinite(S), S, -np.inf), axis=0)
    max_score = np.where(all_nan, np.nan, max_score)
    return pd.DataFrame(
        {
            "stickiness_withinCT_mean": mean_score,
            "stickiness_withinCT_max": max_score,
        }
    )


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Dense rank with 1 = highest score, NaN for missing."""
    s = pd.Series(values, dtype=np.float64)
    return s.rank(method="min", ascending=False).to_numpy()


def stickiness_diagnostics(
    results: pd.DataFrame,
    marker_genes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Quick diagnostics for bias before/after adjustment.

    Returns
    -------
    dict with:
    - `corr_detection_naive_spearman`
    - `corr_detection_resid_spearman`
    - `marker_rank_shift` (DataFrame, if marker genes are supplied)
    """
    out: Dict[str, object] = {}
    if {"detection_rate", "stickiness_naive"}.issubset(results.columns):
        out["corr_detection_naive_spearman"] = (
            results["detection_rate"].corr(results["stickiness_naive"], method="spearman")
        )
    else:
        out["corr_detection_naive_spearman"] = np.nan

    if {"detection_rate", "stickiness_resid"}.issubset(results.columns):
        out["corr_detection_resid_spearman"] = (
            results["detection_rate"].corr(results["stickiness_resid"], method="spearman")
        )
    else:
        out["corr_detection_resid_spearman"] = np.nan

    if marker_genes is not None and len(marker_genes) > 0:
        have = results["gene"].isin(marker_genes)
        cols = ["gene", "rank_naive", "rank_resid", "rank_delta"]
        if set(cols).issubset(results.columns):
            out["marker_rank_shift"] = (
                results.loc[have, cols].sort_values("rank_delta", ascending=False).reset_index(drop=True)
            )
        else:
            out["marker_rank_shift"] = pd.DataFrame(columns=cols)
    return out


def stickiness(
    adata,
    layer: Optional[str] = None,
    cell_type_key: str = "cell_type",
    total_counts_key: str = "total_counts",
    connectivities_key: str = "connectivities",
    key_added: str = "stickiness",
    n_perm: int = 200,
    random_state: Optional[int] = 0,
    compute_within_cell_type: bool = True,
    min_cells_per_type: int = 50,
    store_residuals: bool = False,
    copy: bool = False,
    eps: float = 1e-8,
) -> pd.DataFrame | Tuple[pd.DataFrame, object]:
    """Compute cell-type-adjusted spatial stickiness.

    Parameters
    ----------
    adata
        AnnData object with expression, graph, and metadata.
    layer
        Input layer name. If None, uses `adata.X`.
    cell_type_key
        `adata.obs` key with cell-type labels.
    total_counts_key
        `adata.obs` key with per-cell library sizes.
    connectivities_key
        `adata.obsp` key for spatial connectivities (sparse graph).
    key_added
        Key prefix used for AnnData storage.
    n_perm
        Number of conditional permutations. Set to `0` to skip null calibration.
    random_state
        Seed for permutation RNG.
    compute_within_cell_type
        If True, compute within-cell-type aggregated stickiness columns.
    min_cells_per_type
        Minimum cells per cell type for within-cell-type stickiness.
    store_residuals
        If True, store residuals in `adata.layers[f"{key_added}_resid"]`.
    copy
        If True, compute on a copied AnnData and return `(results_df, adata_copy)`.
    eps
        Numerical stability constant.

    Returns
    -------
    pandas.DataFrame
        Gene-level results (always returned). If `copy=True`, returns
        `(results_df, adata_copy)`.
    """
    if n_perm < 0:
        raise ValueError("n_perm must be >= 0.")
    if min_cells_per_type <= 0:
        raise ValueError("min_cells_per_type must be > 0.")

    adata_work = adata.copy() if copy else adata

    resid_res = _residualize_by_cell_type(
        adata=adata_work,
        layer=layer,
        cell_type_key=cell_type_key,
        total_counts_key=total_counts_key,
        key_added=key_added,
        store_residuals=store_residuals,
        scale_factor=1e4,
        chunk_size=512,
    )
    R = resid_res.residuals
    Y = resid_res.normalized_log_expr
    W = _get_connectivities(adata_work, connectivities_key=connectivities_key)

    # Adjusted stickiness is the metric on residuals.
    score_resid, _, _ = _laplacian_stickiness(R, W=W, eps=eps)
    df = pd.DataFrame(
        {
            "gene": adata_work.var_names.astype(str),
            "stickiness_resid": score_resid,
        }
    )

    # Unadjusted score for diagnostics/rank-shift checks.
    score_naive, _, _ = _laplacian_stickiness(Y, W=W, eps=eps)
    df["stickiness_naive"] = score_naive

    # Conditional permutation null (within-cell-type shuffling).
    if n_perm > 0:
        null_df = permutation_null_stickiness(
            residuals=R,
            observed_scores=score_resid,
            W=W,
            cell_type=resid_res.cell_type,
            log_total_counts=resid_res.log_total_counts,
            n_perm=n_perm,
            random_state=random_state,
            umi_n_bins=None,
            eps=eps,
        )
    else:
        n_genes = adata_work.n_vars
        null_df = pd.DataFrame(
            {
                "null_mean": np.full(n_genes, np.nan),
                "null_sd": np.full(n_genes, np.nan),
                "z": np.full(n_genes, np.nan),
                "pval": np.full(n_genes, np.nan),
                "qval": np.full(n_genes, np.nan),
            }
        )
    df = pd.concat([df, null_df], axis=1)

    # Optional within-cell-type stickiness aggregation.
    if compute_within_cell_type:
        within_df = within_celltype_stickiness(
            residuals=R,
            W=W,
            cell_type=resid_res.cell_type,
            min_cells=min_cells_per_type,
            eps=eps,
        ).rename(
            columns={
                "stickiness_withinCT_mean": "withinCT_mean",
                "stickiness_withinCT_max": "withinCT_max",
            }
        )
        df = pd.concat([df, within_df], axis=1)

    # Diagnostics-ready columns.
    mat_in = _get_matrix(adata_work, layer=layer)
    if sparse.issparse(mat_in):
        detect = np.asarray((mat_in > 0).sum(axis=0)).ravel() / max(1, adata_work.n_obs)
    else:
        detect = np.mean(np.asarray(mat_in) > 0, axis=0).ravel()
    df["detection_rate"] = np.asarray(detect, dtype=np.float64)
    df["rank_naive"] = _rank_desc(df["stickiness_naive"].to_numpy())
    df["rank_resid"] = _rank_desc(df["stickiness_resid"].to_numpy())
    df["rank_delta"] = df["rank_resid"] - df["rank_naive"]

    # Store results in AnnData (`varm` + `uns`) with fixed gene order.
    ordered = df.set_index("gene").loc[adata_work.var_names.astype(str)]
    adata_work.varm[key_added] = ordered.to_records(index=False)
    adata_work.uns[key_added] = {
        "layer": layer,
        "cell_type_key": cell_type_key,
        "total_counts_key": total_counts_key,
        "connectivities_key": connectivities_key,
        "key_added": key_added,
        "n_perm": int(n_perm),
        "random_state": random_state,
        "compute_within_cell_type": bool(compute_within_cell_type),
        "min_cells_per_type": int(min_cells_per_type),
        "store_residuals": bool(store_residuals),
        "eps": float(eps),
        "scale_factor": float(1e4),
        "chunk_size": int(512),
        "columns": list(ordered.columns),
    }

    if copy:
        return df, adata_work
    return df


def compute_celltype_adjusted_stickiness(
    adata,
    layer: Optional[str] = None,
    cell_type_key: str = "cell_type",
    total_counts_key: str = "total_counts",
    connectivities_key: str = "connectivities",
    n_perm: int = 200,
    random_state: Optional[int] = 0,
    min_cells: int = 30,
    eps: float = 1e-12,
    umi_n_bins: Optional[int] = None,
    scale_factor: float = 1e4,
    chunk_size: int = 512,
    residual_layer: Optional[str] = "sticky_resid",
    compute_within_ct: bool = True,
) -> pd.DataFrame:
    """Backward-compatible wrapper around :func:`stickiness`."""
    _ = umi_n_bins  # kept for backward compatibility in caller signatures
    store_residuals = residual_layer is not None
    res = stickiness(
        adata=adata,
        layer=layer,
        cell_type_key=cell_type_key,
        total_counts_key=total_counts_key,
        connectivities_key=connectivities_key,
        key_added="stickiness",
        n_perm=n_perm,
        random_state=random_state,
        compute_within_cell_type=compute_within_ct,
        min_cells_per_type=min_cells,
        store_residuals=store_residuals,
        copy=False,
        eps=eps,
    )
    if store_residuals and residual_layer != "stickiness_resid":
        adata.layers[residual_layer] = adata.layers["stickiness_resid"]
    return res


if __name__ == "__main__":
    # Example usage:
    # df = stickiness(adata, n_perm=200, key_added="stickiness")
    # df.sort_values("z", ascending=False).head(30)
    pass
