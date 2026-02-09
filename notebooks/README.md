# Notebook Workflow

This folder contains Jupyter workflows for running `NeighborNorm` spatial sticky-gene analysis on AnnData (`.h5ad`) files.

## 1) Environment Setup

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r notebooks/requirements.txt
```

Optional kernel registration:

```bash
python -m ipykernel install --user --name neighbornorm --display-name "NeighborNorm (.venv)"
```

Launch Jupyter:

```bash
jupyter lab
```

Open `notebooks/spatial-sticky-gene-corrections.ipynb`, set `DATA_PATH`, then run cells top-to-bottom.

## 2) Implemented Workflows

The notebook now includes two complementary workflows.

### A) Spatial sticky-gene discovery and correction

Backed by `spatial_sticky_gene_corrections.py`.

Implemented features:

- Automatic sticky-gene ranking via `discover_spatially_sticky_genes(...)`.
- Sample-aware graph construction using `sample_key="sample_id"` to prevent cross-sample neighbor mixing.
- Ranking controls:
  - `min_samples_eligible_frac`
  - `score_weights` for `neighbor_corr`, `moran_i`, `contrast_mad`, `detect_frac`
- Correction features via `compute_spatial_sticky_gene_corrections(...)`:
  - `{gene}_local_contrast`
  - `{gene}_resid`
  - optional `{gene}_field_resid`
- Sample-tiled plotting in `plot_sticky_gene_views(...)` using:
  - `sample_key="sample_id"`
  - `tile_samples=True`
  This avoids spatial overlap when multiple samples share coordinate ranges.

Stored outputs:

- `adata.uns["sticky_gene_ranking"]`
- `adata.uns["sticky_gene_ranking_top_genes"]`
- `adata.obs` correction columns per selected gene

### B) Cell-type-adjusted stickiness (recommended for ranking)

Backed by `celltype_adjusted_stickiness.py`.

Implemented features:

- Residualization per gene:
  - `y = log1p(1e4 * counts / total_counts)`
  - model: `y ~ 1 + cell_type + log_total_counts`
- Residual-based graph Laplacian stickiness:
  - `stickiness = 1 - (r^T L r) / (r^T D r + eps)`
- Conditional permutation null preserving cell-type structure:
  - shuffle residuals within `cell_type`
  - optional `umi_n_bins` for within-cell-type UMI-stratified shuffling
- Statistical outputs:
  - `null_mean`, `null_sd`, `z`, `pval`, `qval` (BH-FDR)
- Optional within-cell-type stickiness aggregation:
  - `stickiness_withinCT_mean`
  - `stickiness_withinCT_max`
- Diagnostics helper:
  - detection-rate correlation before vs after residualization
  - marker rank shift table (`rank_naive`, `rank_resid`, `rank_delta`)

Stored outputs:

- `adata.layers["sticky_resid"]` (if `residual_layer` is set)
- DataFrame `ctas_df` with ranking/statistical/diagnostic columns

## 3) Key Data Requirements

- `adata.obsm["spatial"]` is required for spatial-neighbor workflows.
- `adata.obs["sample_id"]` is used for sample-aware neighbor building and tiled plotting.
- `adata.obs["cell_type"]` is required for cell-type-adjusted stickiness.
- `adata.obs["total_counts"]` is used for normalization and residualization; if missing, it is computed from the selected matrix.
- If `adata.layers["counts"]` is missing, the workflow can fallback to `adata.X`.

## 4) Practical Notes

- The notebook reloads modules in-place (`importlib.reload(...)`) to avoid stale imports during iterative development.
- For the cell-type-adjusted permutation null, `n_perm=200` is a good default; increase for more stable p-values.
- If runtime is high, reduce genes, reduce `n_perm`, or tighten filtering thresholds before ranking.
