# NeighborNorm

Spatial sticky-gene analysis utilities for AnnData, with support for:

- sample-aware spatial neighbor handling
- automatic sticky-gene discovery
- correction feature generation for selected genes
- cell-type-adjusted stickiness ranking with permutation-based significance

## Repository Structure

- `spatial_sticky_gene_corrections.py`
  Spatial sticky-gene discovery, correction features, and plotting helpers.
- `celltype_adjusted_stickiness.py`
  Cell-type-adjusted residual stickiness ranking with conditional permutation null.
- `notebooks/spatial-sticky-gene-corrections.ipynb`
  End-to-end interactive workflow.
- `notebooks/README.md`
  Notebook-focused instructions and details.

## Setup

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

## Data Requirements

Minimum required for most workflows:

- AnnData with expression in `adata.X` or a chosen layer (often `adata.layers["counts"]`)
- spatial coordinates in `adata.obsm["spatial"]`
- spatial graph in `adata.obsp["connectivities"]` (or module-generated key)

Recommended metadata:

- `adata.obs["sample_id"]` for sample-aware graph construction and plotting
- `adata.obs["cell_type"]` for cell-type-adjusted stickiness
- `adata.obs["total_counts"]` (auto-computed when missing in CT-adjusted workflow)

## Workflow A: Spatial Sticky-Gene Discovery + Corrections

Main module: `spatial_sticky_gene_corrections.py`

### Discover sticky genes

```python
from spatial_sticky_gene_corrections import discover_spatially_sticky_genes

ranking = discover_spatially_sticky_genes(
    adata,
    n_neighbors=15,
    key="spatial_neighbors",
    sample_key="sample_id",  # sample-aware neighbors
    layer_counts="counts",
    min_samples_eligible_frac=1.0,
    score_weights={
        "neighbor_corr": 0.45,
        "moran_i": 0.45,
        "contrast_mad": 0.10,
        "detect_frac": 0.0,
    },
    top_k=20,
)
```

Stored in:

- `adata.uns["sticky_gene_ranking"]`
- `adata.uns["sticky_gene_ranking_top_genes"]`

### Compute correction features

```python
from spatial_sticky_gene_corrections import compute_spatial_sticky_gene_corrections

adata = compute_spatial_sticky_gene_corrections(
    adata,
    sticky_genes=["MBP", "GFAP"],
    n_neighbors=15,
    key="spatial_neighbors",
    sample_key="sample_id",
    layer_counts="counts",
    do_field_residual=True,
)
```

Creates per-gene columns in `adata.obs`:

- `{gene}_local_contrast`
- `{gene}_resid`
- optional `{gene}_field_resid`

### Plot without sample overlap

```python
from spatial_sticky_gene_corrections import plot_sticky_gene_views

plot_sticky_gene_views(
    adata,
    genes=["MBP", "GFAP"],
    sample_key="sample_id",
    tile_samples=True,
)
```

## Workflow B: Cell-Type-Adjusted Stickiness (Recommended Ranking)

Main module: `celltype_adjusted_stickiness.py`

This workflow residualizes expression by cell type + library size before
spatial smoothness scoring, then computes a conditional permutation null.

```python
from celltype_adjusted_stickiness import (
    compute_celltype_adjusted_stickiness,
    stickiness_diagnostics,
)

df = compute_celltype_adjusted_stickiness(
    adata,
    layer="counts" if "counts" in adata.layers else None,
    cell_type_key="cell_type",
    total_counts_key="total_counts",
    connectivities_key="connectivities",
    n_perm=200,
    random_state=0,
    min_cells=30,
    umi_n_bins=4,
    residual_layer="sticky_resid",
    compute_within_ct=True,
)

top = df.sort_values("z", ascending=False).head(30)
diag = stickiness_diagnostics(df, marker_genes=["COL4A1", "MBP", "GFAP"])
```

Key columns in returned DataFrame:

- `gene`
- `stickiness_raw` / `stickiness_resid`
- `null_mean`, `null_sd`, `z`, `pval`, `qval`
- optional `stickiness_withinCT_mean`, `stickiness_withinCT_max`
- diagnostics: `detection_rate`, `stickiness_naive`, `rank_naive`, `rank_resid`, `rank_delta`

Also writes residuals to:

- `adata.layers["sticky_resid"]` (if `residual_layer` is not `None`)

## Notebook

Use `notebooks/spatial-sticky-gene-corrections.ipynb` for an end-to-end run.
It includes:

- spatial sticky-gene discovery/correction workflow
- cell-type-adjusted stickiness workflow
- diagnostics for marker rank changes and detection-rate correlation shifts

Detailed notebook instructions are in `notebooks/README.md`.

