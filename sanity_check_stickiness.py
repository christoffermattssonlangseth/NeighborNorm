"""Minimal sanity check for cell-type-adjusted stickiness.

Usage
-----
python sanity_check_stickiness.py /path/to/data.h5ad \
  --layer counts \
  --cell-type-key cell_type \
  --markers COL4A1,MBP,GFAP
"""

from __future__ import annotations

import argparse

from anndata import read_h5ad

from celltype_adjusted_stickiness import stickiness, stickiness_diagnostics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5ad_path", help="Input AnnData file.")
    p.add_argument("--layer", default=None, help="Expression layer (default: adata.X).")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--total-counts-key", default="total_counts")
    p.add_argument("--connectivities-key", default="connectivities")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--min-cells-per-type", type=int, default=50)
    p.add_argument(
        "--markers",
        default="COL4A1,MBP,GFAP",
        help="Comma-separated marker genes to track rank shifts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    markers = [m.strip() for m in args.markers.split(",") if m.strip()]

    adata = read_h5ad(args.h5ad_path)
    df = stickiness(
        adata,
        layer=args.layer,
        cell_type_key=args.cell_type_key,
        total_counts_key=args.total_counts_key,
        connectivities_key=args.connectivities_key,
        key_added="stickiness",
        n_perm=args.n_perm,
        random_state=args.random_state,
        compute_within_cell_type=True,
        min_cells_per_type=args.min_cells_per_type,
        store_residuals=False,
        copy=False,
    )
    diag = stickiness_diagnostics(df, marker_genes=markers)

    print("Top genes by z:")
    print(df.sort_values("z", ascending=False).head(20)[["gene", "stickiness_resid", "z", "qval"]])
    print()
    print("Detection-rate correlation (naive):", diag["corr_detection_naive_spearman"])
    print("Detection-rate correlation (residual):", diag["corr_detection_resid_spearman"])
    print()
    print("Marker rank shifts (positive rank_delta means marker dropped after adjustment):")
    print(diag["marker_rank_shift"])


if __name__ == "__main__":
    main()

