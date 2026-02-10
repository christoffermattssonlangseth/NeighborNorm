# NeighborNorm

![NeighborNorm logo](assets/logo.png)

Spatial sticky-gene analysis utilities for AnnData, with support for:

- sample-aware spatial neighbor handling
- automatic sticky-gene discovery
- correction feature generation for selected genes
- cell-type-adjusted stickiness ranking with permutation-based significance
- gene-level leakage / mis-segmentation scoring

## Repository Structure

- `spatial_sticky_gene_corrections.py`
  Spatial sticky-gene discovery, correction features, and plotting helpers.
- `celltype_adjusted_stickiness.py`
  Cell-type-adjusted residual stickiness ranking with conditional permutation null.
- `leakage_score.py`
  Gene-level leakage/mis-segmentation metrics and combined leakage score.
- `tl.py`
  Tool-style API facade exposing `tl.stickiness` and `tl.leakage_score`.
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
    stickiness,
    stickiness_diagnostics,
)

df = stickiness(
    adata,
    layer="counts" if "counts" in adata.layers else None,
    cell_type_key="cell_type",
    total_counts_key="total_counts",
    connectivities_key="connectivities",
    key_added="stickiness",
    n_perm=200,
    random_state=0,
    compute_within_cell_type=True,
    min_cells_per_type=50,
    store_residuals=False,
    copy=False,
    show_progress=True,  # tqdm-style progress if tqdm is installed
)

top = df.sort_values("z", ascending=False).head(30)
diag = stickiness_diagnostics(df, marker_genes=["COL4A1", "MBP", "GFAP"])
```

Key columns in returned DataFrame:

- `gene`
- `stickiness_raw` / `stickiness_resid`
- `null_mean`, `null_sd`, `z`, `pval`, `qval`
- optional `withinCT_mean`, `withinCT_max`
- diagnostics: `detection_rate`, `stickiness_naive`, `rank_naive`, `rank_resid`, `rank_delta`

Stored in AnnData:

- `adata.varm["stickiness"]` (structured array of per-gene stats)
- `adata.uns["stickiness"]` (run parameters)
- optional `adata.layers["stickiness_resid"]` when `store_residuals=True`

## Workflow C: Leakage / Mis-Segmentation Scoring

Main modules: `leakage_score.py` and `tl.py`

This workflow scores genes that show broad low-count spillover outside expected
cell types, anchored to a source cell type for each gene. It is not a spatial
smoothness metric.

```python
import tl

df = tl.leakage_score(
    adata,
    layer="counts" if "counts" in adata.layers else None,
    cell_type_key="cell_type",
    key_added="leakage",
    specificity_thresh=0.75,
    source_mass_cover=0.95,
    max_sources=1,
    low_count_max=2,
    alpha=2.0,
    pr0=0.02,
    sigma=0.8,
    connectivities_key="connectivities",
)
df.head(30)
```

Key columns in returned DataFrame:

- `gene`
- `top1_ct`, `top1_frac`, `sources`
- `off_type_frac`
- `spray_off`, `spray_delta`
- `presence_ratio`
- `pr_score`
- `near_enrichment`
- optional `mean_ratio`
- `leakage_score`
- `rank`

Stored in AnnData:

- `adata.varm["leakage"]`
- `adata.uns["leakage"]`

## Notebook

Use `notebooks/spatial-sticky-gene-corrections.ipynb` for an end-to-end run.
It includes:

- spatial sticky-gene discovery/correction workflow
- cell-type-adjusted stickiness workflow
- diagnostics for marker rank changes and detection-rate correlation shifts

Detailed notebook instructions are in `notebooks/README.md`.

## Sanity Check Script

Run a quick post-fit sanity check with:

```bash
python sanity_check_stickiness.py /path/to/data.h5ad --layer counts --markers COL4A1,MBP,GFAP
```

The script reports:

- top genes by adjusted z-score
- detection-rate correlation before vs after adjustment
- marker rank shifts (`rank_delta`)

Leakage sanity check:

```bash
python sanity_check_leakage.py /path/to/data.h5ad --layer counts --markers GFAP,MBP
```

The script reports:

- top genes by `leakage_score` with directional columns
- marker ranks for GFAP/MBP/PLP1/AQP4
- housekeeping and high-`presence_ratio` checks in top-ranked genes
