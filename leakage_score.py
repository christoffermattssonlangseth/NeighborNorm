"""Directional leakage / mis-segmentation scoring for AnnData.

This module prioritizes genes that are source-cell-type specific, but show
low-count expression in nearby non-source cells, consistent with directional
mis-assignment.

Example
-------
```python
import tl

df = tl.leakage_score(adata, layer="counts")
df.head(30)
```
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from celltype_adjusted_stickiness import _get_connectivities, _get_matrix


def _to_2d_float(mat) -> np.ndarray:
    """Convert sparse/dense matrix-like input to dense float64 array."""
    if sparse.issparse(mat):
        return mat.toarray().astype(np.float64, copy=False)
    return np.asarray(mat, dtype=np.float64)


def _robust_zscore(
    values: np.ndarray,
    eps: float = 1e-8,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Robust z-score with median/MAD, restricted to optional valid mask."""
    arr = np.asarray(values, dtype=np.float64)
    z = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if valid.sum() == 0:
        return z

    x = arr[valid]
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale <= eps:
        sd = np.std(x)
        if sd <= eps:
            z[valid] = 0.0
            return z
        z[valid] = (x - np.mean(x)) / (sd + eps)
        return z

    z[valid] = (x - med) / (scale + eps)
    return z


def _validate_inputs(
    adata,
    cell_type_key: str,
    specificity_thresh: float,
    source_mass_cover: float,
    max_sources: int,
    low_count_max: int,
    alpha: float,
    pr0: float,
    sigma: float,
    chunk_size: int,
    eps: float,
) -> None:
    if cell_type_key not in adata.obs:
        raise KeyError(f"'{cell_type_key}' not found in adata.obs.")
    if not (0.0 < float(specificity_thresh) <= 1.0):
        raise ValueError("specificity_thresh must be in (0, 1].")
    if not (0.0 < float(source_mass_cover) <= 1.0):
        raise ValueError("source_mass_cover must be in (0, 1].")
    if int(max_sources) < 1:
        raise ValueError("max_sources must be >= 1.")
    if int(low_count_max) < 1:
        raise ValueError("low_count_max must be >= 1.")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be > 0.")
    if float(pr0) <= 0.0:
        raise ValueError("pr0 must be > 0.")
    if float(sigma) <= 0.0:
        raise ValueError("sigma must be > 0.")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1.")
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0.")


def _build_celltype_design(
    cell_type: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-type codes, names, and per-type cell counts."""
    ct_cat = pd.Categorical(pd.Series(cell_type, dtype="string"))
    codes = ct_cat.codes.astype(np.int64, copy=False)
    names = ct_cat.categories.astype(str).to_numpy()
    counts = np.bincount(codes, minlength=names.size).astype(np.float64)
    return codes, names, counts


def _build_membership(cell_type_codes: np.ndarray, n_celltypes: int) -> sparse.csr_matrix:
    """Cell-by-celltype one-hot membership matrix."""
    n_cells = cell_type_codes.shape[0]
    return sparse.csr_matrix(
        (
            np.ones(n_cells, dtype=np.float64),
            (np.arange(n_cells, dtype=np.int64), cell_type_codes),
        ),
        shape=(n_cells, n_celltypes),
    )


def _aggregate_mass_by_celltype(
    mat,
    membership: sparse.csr_matrix,
) -> np.ndarray:
    """Return cell-type x gene expression mass matrix."""
    return _to_2d_float(membership.T @ mat)


def _aggregate_detection_and_low_by_celltype(
    mat,
    membership: sparse.csr_matrix,
    low_count_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cell-type x gene nonzero and low-count detection matrices."""
    if sparse.issparse(mat):
        mat_csr = mat.tocsr(copy=True)

        nz = mat_csr.copy()
        nz.data = (nz.data > 0.0).astype(np.float64)
        nz.eliminate_zeros()
        ct_nz = _to_2d_float(membership.T @ nz)

        low = mat_csr.copy()
        low.data = ((low.data >= 1.0) & (low.data <= float(low_count_max))).astype(np.float64)
        low.eliminate_zeros()
        ct_low = _to_2d_float(membership.T @ low)
        return ct_nz, ct_low

    arr = np.asarray(mat, dtype=np.float64)
    ct_nz = _to_2d_float(membership.T @ (arr > 0.0).astype(np.float64))
    ct_low = _to_2d_float(
        membership.T @ ((arr >= 1.0) & (arr <= float(low_count_max))).astype(np.float64)
    )
    return ct_nz, ct_low


def _select_sources(
    frac: np.ndarray,
    source_mass_cover: float,
    max_sources: int,
) -> np.ndarray:
    """Build source cell-type mask per gene from mass fractions."""
    n_celltypes, n_genes = frac.shape
    if n_celltypes == 0 or n_genes == 0:
        return np.zeros((n_celltypes, n_genes), dtype=bool)

    order = np.argsort(-frac, axis=0)
    sorted_frac = np.take_along_axis(frac, order, axis=0)
    cum = np.cumsum(sorted_frac, axis=0)
    n_needed = (cum < float(source_mass_cover)).sum(axis=0) + 1
    n_needed = np.clip(n_needed, 1, int(max_sources))

    row_ids = np.arange(n_celltypes, dtype=np.int64)[:, None]
    selected_sorted = (row_ids < n_needed[None, :]) & (sorted_frac > 0.0)

    mask = np.zeros((n_celltypes, n_genes), dtype=bool)
    col_ids = np.tile(np.arange(n_genes, dtype=np.int64), n_celltypes)
    mask[order.ravel(), col_ids] = selected_sorted.ravel()
    return mask


def _format_sources(source_mask: np.ndarray, celltype_names: np.ndarray) -> np.ndarray:
    """Serialize per-gene source cell types as ';'-joined strings."""
    n_genes = source_mask.shape[1]
    out = np.empty(n_genes, dtype=object)
    for g in range(n_genes):
        idx = np.where(source_mask[:, g])[0]
        out[g] = ";".join(celltype_names[idx]) if idx.size > 0 else ""
    return out


def _compute_near_any(W, on_mask: np.ndarray) -> np.ndarray:
    """Return per-cell indicator of having any 1-hop ON neighbor."""
    if sparse.issparse(W):
        return (np.asarray(W[:, on_mask].sum(axis=1)).ravel() > 0).astype(bool, copy=False)
    dense = np.asarray(W, dtype=np.float64)
    return (np.sum(dense[:, on_mask], axis=1) > 0.0).astype(bool, copy=False)


def _compute_neighbor_metrics_chunked(
    mat,
    W,
    cell_type_codes: np.ndarray,
    source_mask: np.ndarray,
    eligible: np.ndarray,
    low_count_max: int,
    eps: float,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute OFF+ neighbor proximity and proximity-aware spray metrics per gene."""
    n_genes = source_mask.shape[1]
    near_frac = np.full(n_genes, np.nan, dtype=np.float64)
    near_enrichment = np.full(n_genes, np.nan, dtype=np.float64)
    spray_near = np.full(n_genes, np.nan, dtype=np.float64)
    spray_far = np.full(n_genes, np.nan, dtype=np.float64)
    spray_delta = np.full(n_genes, np.nan, dtype=np.float64)

    source_cache: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    if sparse.issparse(mat):
        mat_csc = mat.tocsc(copy=False)
        indptr = mat_csc.indptr
        indices = mat_csc.indices
        data = mat_csc.data

        for start in range(0, n_genes, int(chunk_size)):
            end = min(start + int(chunk_size), n_genes)
            for g in range(start, end):
                if not bool(eligible[g]):
                    continue

                src_idx = np.where(source_mask[:, g])[0]
                if src_idx.size == 0:
                    continue
                key = tuple(src_idx.tolist())

                if key not in source_cache:
                    on_mask = np.isin(cell_type_codes, np.asarray(key, dtype=np.int64))
                    off_mask = ~on_mask
                    near_any = _compute_near_any(W, on_mask=on_mask)
                    baseline = float(np.mean(near_any[off_mask])) if off_mask.any() else np.nan
                    source_cache[key] = (on_mask, off_mask, near_any, baseline)
                _, off_mask, near_any, baseline = source_cache[key]

                c0, c1 = indptr[g], indptr[g + 1]
                rows = indices[c0:c1]
                vals = data[c0:c1]
                if vals.size == 0:
                    off_rows = np.empty(0, dtype=np.int64)
                    off_vals = np.empty(0, dtype=np.float64)
                else:
                    pos = vals > 0.0
                    rows = rows[pos]
                    vals = vals[pos]
                    off_sel = off_mask[rows]
                    off_rows = rows[off_sel]
                    off_vals = vals[off_sel]

                n_off_pos = int(off_rows.size)
                if n_off_pos == 0:
                    near_frac[g] = 0.0
                    near_enrichment[g] = 0.0
                    spray_near[g] = 0.0
                    spray_far[g] = 0.0
                    spray_delta[g] = 0.0
                    continue

                near_flags = near_any[off_rows]
                near_count = int(np.sum(near_flags))
                near_frac[g] = near_count / (float(n_off_pos) + float(eps))
                if np.isfinite(baseline):
                    near_enrichment[g] = near_frac[g] / (baseline + float(eps))

                low_flags = (off_vals >= 1.0) & (off_vals <= float(low_count_max))
                if near_count > 0:
                    spray_near[g] = float(np.mean(low_flags[near_flags]))
                else:
                    spray_near[g] = 0.0

                far_count = n_off_pos - near_count
                if far_count > 0:
                    spray_far[g] = float(np.mean(low_flags[~near_flags]))
                else:
                    spray_far[g] = 0.0

                spray_delta[g] = spray_near[g] - spray_far[g]

        return near_frac, near_enrichment, spray_near, spray_far, spray_delta

    dense = np.asarray(mat, dtype=np.float64)
    for start in range(0, n_genes, int(chunk_size)):
        end = min(start + int(chunk_size), n_genes)
        block = dense[:, start:end]

        for local_idx, g in enumerate(range(start, end)):
            if not bool(eligible[g]):
                continue

            src_idx = np.where(source_mask[:, g])[0]
            if src_idx.size == 0:
                continue
            key = tuple(src_idx.tolist())

            if key not in source_cache:
                on_mask = np.isin(cell_type_codes, np.asarray(key, dtype=np.int64))
                off_mask = ~on_mask
                near_any = _compute_near_any(W, on_mask=on_mask)
                baseline = float(np.mean(near_any[off_mask])) if off_mask.any() else np.nan
                source_cache[key] = (on_mask, off_mask, near_any, baseline)
            _, off_mask, near_any, baseline = source_cache[key]

            col = block[:, local_idx]
            off_rows = np.where(off_mask & (col > 0.0))[0]
            off_vals = col[off_rows]
            n_off_pos = int(off_rows.size)
            if n_off_pos == 0:
                near_frac[g] = 0.0
                near_enrichment[g] = 0.0
                spray_near[g] = 0.0
                spray_far[g] = 0.0
                spray_delta[g] = 0.0
                continue

            near_flags = near_any[off_rows]
            near_count = int(np.sum(near_flags))
            near_frac[g] = near_count / (float(n_off_pos) + float(eps))
            if np.isfinite(baseline):
                near_enrichment[g] = near_frac[g] / (baseline + float(eps))

            low_flags = (off_vals >= 1.0) & (off_vals <= float(low_count_max))
            if near_count > 0:
                spray_near[g] = float(np.mean(low_flags[near_flags]))
            else:
                spray_near[g] = 0.0

            far_count = n_off_pos - near_count
            if far_count > 0:
                spray_far[g] = float(np.mean(low_flags[~near_flags]))
            else:
                spray_far[g] = 0.0

            spray_delta[g] = spray_near[g] - spray_far[g]

    return near_frac, near_enrichment, spray_near, spray_far, spray_delta


def leakage_score(
    adata,
    layer: Optional[str] = None,
    cell_type_key: str = "cell_type",
    key_added: str = "leakage",
    specificity_thresh: float = 0.75,
    source_mass_cover: float = 0.95,
    max_sources: int = 1,
    low_count_max: int = 2,
    alpha: float = 2.0,
    copy: bool = False,
    eps: float = 1e-8,
    pr0: float = 0.02,
    sigma: float = 0.8,
    connectivities_key: str = "connectivities",
    chunk_size: int = 512,
) -> pd.DataFrame | Tuple[pd.DataFrame, object]:
    """Compute directional gene-level leakage / mis-segmentation score.

    Parameters
    ----------
    adata
        AnnData object.
    layer
        Layer for counts. If None, uses `adata.X`.
    cell_type_key
        `adata.obs` key containing cell-type labels.
    key_added
        Storage key used in `adata.varm` and `adata.uns`.
    specificity_thresh
        Minimum top-1 cell-type mass fraction required for eligibility.
    source_mass_cover
        Cumulative mass-fraction target used to define source cell types.
    max_sources
        Maximum number of source cell types per gene. Defaults to 1 for
        directional leakage.
    low_count_max
        Maximum count used for low-count spray detection (inclusive).
    alpha
        Specificity weighting exponent in final score (`top1_frac ** alpha`).
    copy
        If True, run on a copied AnnData and return `(results_df, adata_copy)`.
    eps
        Numerical stability constant.
    pr0
        Target presence-ratio center for leakage band-pass score.
    sigma
        Log-space width for leakage band-pass score.
    connectivities_key
        Graph key in `adata.obsp` used for neighborhood enrichment.
    chunk_size
        Number of genes processed per chunk for neighbor-based metrics.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, AnnData)
        Results sorted by `leakage_score` descending. If `copy=True`, returns
        `(results_df, adata_copy)`.
    """
    _validate_inputs(
        adata=adata,
        cell_type_key=cell_type_key,
        specificity_thresh=specificity_thresh,
        source_mass_cover=source_mass_cover,
        max_sources=max_sources,
        low_count_max=low_count_max,
        alpha=alpha,
        pr0=pr0,
        sigma=sigma,
        chunk_size=chunk_size,
        eps=eps,
    )

    adata_work = adata.copy() if copy else adata
    mat = _get_matrix(adata_work, layer=layer)
    W = _get_connectivities(adata_work, connectivities_key=connectivities_key)

    n_cells = int(adata_work.n_obs)
    n_genes = int(adata_work.n_vars)

    cell_type = (
        adata_work.obs[cell_type_key]
        .astype("string")
        .fillna("__missing_cell_type__")
        .to_numpy()
    )
    ct_codes, ct_names, ct_n = _build_celltype_design(cell_type=cell_type)
    membership = _build_membership(cell_type_codes=ct_codes, n_celltypes=ct_names.size)

    ct_mass = _aggregate_mass_by_celltype(mat=mat, membership=membership)
    total_mass = np.sum(ct_mass, axis=0, dtype=np.float64)
    frac = np.divide(
        ct_mass,
        total_mass[None, :] + float(eps),
        out=np.zeros_like(ct_mass, dtype=np.float64),
        where=True,
    )

    gene_idx = np.arange(n_genes, dtype=np.int64)
    top1_idx = np.argmax(frac, axis=0)
    top1_frac = frac[top1_idx, gene_idx]
    top1_ct = ct_names[top1_idx]
    entropy = -np.sum(np.where(frac > 0.0, frac * np.log(frac + float(eps)), 0.0), axis=0)

    eligible = (top1_frac >= float(specificity_thresh)) & (total_mass > 0.0)

    source_mask = _select_sources(
        frac=frac,
        source_mass_cover=source_mass_cover,
        max_sources=max_sources,
    )
    source_n = source_mask.sum(axis=0)

    # Ensure every eligible gene has at least one source type.
    missing_source = eligible & (source_n == 0)
    if np.any(missing_source):
        miss_idx = np.where(missing_source)[0]
        source_mask[top1_idx[miss_idx], miss_idx] = True
        source_n = source_mask.sum(axis=0)

    sources = _format_sources(source_mask=source_mask, celltype_names=ct_names)

    ct_nz, ct_low = _aggregate_detection_and_low_by_celltype(
        mat=mat,
        membership=membership,
        low_count_max=low_count_max,
    )
    source_float = source_mask.astype(np.float64)

    on_cells = np.sum(source_float * ct_n[:, None], axis=0, dtype=np.float64)
    off_cells = np.maximum(float(n_cells) - on_cells, 0.0)

    on_mass = np.sum(source_float * ct_mass, axis=0, dtype=np.float64)
    off_mass = np.maximum(total_mass - on_mass, 0.0)

    total_nz = np.sum(ct_nz, axis=0, dtype=np.float64)
    on_nz = np.sum(source_float * ct_nz, axis=0, dtype=np.float64)
    off_nz = np.maximum(total_nz - on_nz, 0.0)

    total_low = np.sum(ct_low, axis=0, dtype=np.float64)
    on_low = np.sum(source_float * ct_low, axis=0, dtype=np.float64)
    off_low = np.maximum(total_low - on_low, 0.0)

    p_on = on_nz / (on_cells + float(eps))
    p_off = off_nz / (off_cells + float(eps))
    mean_on = on_mass / (on_cells + float(eps))
    mean_off = off_mass / (off_cells + float(eps))

    off_type_frac = off_mass / (total_mass + float(eps))
    spray_off = off_low / (off_nz + float(eps))
    presence_ratio = p_off / (p_on + float(eps))
    mean_ratio = mean_off / (mean_on + float(eps))

    log_pr = np.log(presence_ratio + float(eps))
    pr_score = np.exp(
        -((log_pr - np.log(float(pr0))) ** 2) / (2.0 * (float(sigma) ** 2))
    ).astype(np.float64, copy=False)

    near_frac, near_enrichment, spray_near, spray_far, spray_delta = _compute_neighbor_metrics_chunked(
        mat=mat,
        W=W,
        cell_type_codes=ct_codes,
        source_mask=source_mask,
        eligible=eligible,
        low_count_max=low_count_max,
        eps=float(eps),
        chunk_size=int(chunk_size),
    )

    # Ineligible genes are removed from directional leakage ranking.
    for arr in (
        off_type_frac,
        spray_off,
        presence_ratio,
        mean_ratio,
        pr_score,
        near_frac,
        near_enrichment,
        spray_near,
        spray_far,
        spray_delta,
    ):
        arr[~eligible] = np.nan

    z_off_type_frac = _robust_zscore(off_type_frac, eps=float(eps), valid_mask=eligible)
    z_near_enrichment = _robust_zscore(near_enrichment, eps=float(eps), valid_mask=eligible)
    z_spray_delta = _robust_zscore(spray_delta, eps=float(eps), valid_mask=eligible)
    z_pr_score = _robust_zscore(pr_score, eps=float(eps), valid_mask=eligible)

    z_stack = np.column_stack([z_off_type_frac, z_near_enrichment, z_spray_delta, z_pr_score])
    base_score = np.nansum(z_stack, axis=1)
    base_score[np.all(~np.isfinite(z_stack), axis=1)] = np.nan

    specificity_weight = np.power(np.clip(top1_frac, 0.0, 1.0), float(alpha))
    leakage = base_score * specificity_weight
    leakage[~eligible] = np.nan

    rank = pd.Series(leakage, dtype=np.float64).rank(method="min", ascending=False).to_numpy()

    base = pd.DataFrame(
        {
            "gene": adata_work.var_names.astype(str),
            "leakage_score": leakage,
            "top1_ct": top1_ct,
            "top1_frac": top1_frac,
            "sources": sources,
            "off_type_frac": off_type_frac,
            "spray_off": spray_off,
            "presence_ratio": presence_ratio,
            "mean_ratio": mean_ratio,
            "near_frac": near_frac,
            "near_enrichment": near_enrichment,
            "spray_near": spray_near,
            "spray_far": spray_far,
            "spray_delta": spray_delta,
            "pr_score": pr_score,
            "rank": rank,
            "eligible": eligible.astype(bool),
            "source_n": source_n.astype(np.int64, copy=False),
            "entropy": entropy,
            "p_on": p_on,
            "p_off": p_off,
            "mean_on": mean_on,
            "mean_off": mean_off,
            "z_off_type_frac": z_off_type_frac,
            "z_near_enrichment": z_near_enrichment,
            "z_spray_delta": z_spray_delta,
            "z_pr_score": z_pr_score,
            "base_score": base_score,
            "specificity_weight": specificity_weight,
        }
    )

    ordered = base.set_index("gene").loc[adata_work.var_names.astype(str)]
    adata_work.varm[key_added] = ordered.to_records(index=False)
    adata_work.uns[key_added] = {
        "description": (
            "Directional leakage score: source-specific genes with OFF-cell "
            "near-source enrichment and low-count halo are prioritized."
        ),
        "params": {
            "layer": layer,
            "cell_type_key": cell_type_key,
            "key_added": key_added,
            "specificity_thresh": float(specificity_thresh),
            "source_mass_cover": float(source_mass_cover),
            "max_sources": int(max_sources),
            "low_count_max": int(low_count_max),
            "alpha": float(alpha),
            "pr0": float(pr0),
            "sigma": float(sigma),
            "connectivities_key": connectivities_key,
            "chunk_size": int(chunk_size),
            "eps": float(eps),
        },
        "columns": list(ordered.columns),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_cell_types": int(ct_names.size),
        "n_eligible_genes": int(np.sum(eligible)),
    }

    result = base.sort_values(by="leakage_score", ascending=False, na_position="last").reset_index(
        drop=True
    )
    if copy:
        return result, adata_work
    return result


def leakage_sanity_check(
    results: pd.DataFrame,
    markers: Sequence[str] = ("GFAP", "MBP", "PLP1", "AQP4"),
    top_n: int = 30,
    housekeeping_prefixes: Sequence[str] = ("MT-", "RPL", "RPS", "EEF1"),
) -> Dict[str, object]:
    """Summarize directional leakage results for quick sanity checks."""
    need = {"gene", "leakage_score"}
    missing = need - set(results.columns)
    if missing:
        raise KeyError(f"results missing required columns: {sorted(missing)}")

    ranked = results.sort_values("leakage_score", ascending=False, na_position="last").copy()
    top = ranked.head(int(top_n)).copy()

    marker_rows = []
    by_gene = ranked.set_index("gene")
    marker_cols = ["rank", "eligible", "top1_frac", "sources", "presence_ratio", "near_enrichment"]
    for marker in markers:
        row_out = {"gene": marker, "in_top_n": False}
        if marker in by_gene.index:
            row = by_gene.loc[marker]
            for col in marker_cols:
                row_out[col] = row[col] if col in by_gene.columns else np.nan
            row_out["in_top_n"] = bool(
                np.isfinite(row_out.get("rank", np.nan))
                and float(row_out.get("rank", np.nan)) <= float(top_n)
            )
        else:
            for col in marker_cols:
                row_out[col] = np.nan if col != "sources" else ""
        marker_rows.append(row_out)
    marker_df = pd.DataFrame(marker_rows)

    prefixes = tuple(p.upper() for p in housekeeping_prefixes)
    hk_mask = top["gene"].astype(str).str.upper().str.startswith(prefixes)
    hk_cols = [c for c in ["gene", "leakage_score", "top1_frac", "eligible", "rank"] if c in top]
    housekeeping_top = top.loc[hk_mask, hk_cols].copy()

    if "presence_ratio" in top.columns:
        mid_mask = top["presence_ratio"].between(0.4, 0.6, inclusive="both")
        mid_cols = [
            c
            for c in [
                "gene",
                "leakage_score",
                "presence_ratio",
                "pr_score",
                "near_enrichment",
                "spray_delta",
                "rank",
            ]
            if c in top.columns
        ]
        mid_presence_top = top.loc[mid_mask, mid_cols].copy()
    else:
        mid_presence_top = pd.DataFrame()

    return {
        "top_genes": top,
        "marker_summary": marker_df,
        "housekeeping_top": housekeeping_top,
        "housekeeping_count_top_n": int(hk_mask.sum()),
        "mid_presence_top": mid_presence_top,
        "mid_presence_count_top_n": int(len(mid_presence_top)),
        "ineligible_count_top_n": int((~top["eligible"].astype(bool)).sum())
        if "eligible" in top
        else np.nan,
    }


def prepare_leakage_scatter(results: pd.DataFrame) -> pd.DataFrame:
    """Return plotting-ready columns for spray-delta vs off-type scatter."""
    need = {"gene", "off_type_frac", "spray_delta", "leakage_score"}
    missing = need - set(results.columns)
    if missing:
        raise KeyError(f"results missing required columns: {sorted(missing)}")
    cols = ["gene", "off_type_frac", "spray_delta", "leakage_score"]
    add = [c for c in ["near_enrichment", "pr_score", "top1_frac", "eligible"] if c in results.columns]
    return results.loc[:, cols + add].copy()

