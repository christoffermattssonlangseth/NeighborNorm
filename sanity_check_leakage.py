"""Minimal sanity check for leakage / mis-segmentation scoring.

Usage
-----
python sanity_check_leakage.py /path/to/data.h5ad \
  --layer counts \
  --cell-type-key cell_type \
  --markers GFAP,MBP
"""

from __future__ import annotations

import argparse

from anndata import read_h5ad

import tl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5ad_path", help="Input AnnData file.")
    p.add_argument("--layer", default=None, help="Expression layer (default: adata.X).")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--key-added", default="leakage")
    p.add_argument("--specificity-thresh", type=float, default=0.75)
    p.add_argument("--source-mass-cover", type=float, default=0.95)
    p.add_argument("--max-sources", type=int, default=1)
    p.add_argument("--low-count-max", type=int, default=2)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--pr0", type=float, default=0.02)
    p.add_argument("--sigma", type=float, default=0.8)
    p.add_argument("--connectivities-key", default="connectivities")
    p.add_argument(
        "--markers",
        default="GFAP,MBP,PLP1,AQP4",
        help="Comma-separated marker genes for quick rank checks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    markers = [m.strip() for m in args.markers.split(",") if m.strip()]

    adata = read_h5ad(args.h5ad_path)
    df = tl.leakage_score(
        adata,
        layer=args.layer,
        cell_type_key=args.cell_type_key,
        key_added=args.key_added,
        specificity_thresh=args.specificity_thresh,
        source_mass_cover=args.source_mass_cover,
        max_sources=args.max_sources,
        low_count_max=args.low_count_max,
        alpha=args.alpha,
        pr0=args.pr0,
        sigma=args.sigma,
        connectivities_key=args.connectivities_key,
        copy=False,
    )

    summary = tl.leakage_sanity_check(df, markers=markers, top_n=30)
    top_cols = [
        c
        for c in [
            "gene",
            "leakage_score",
            "top1_ct",
            "top1_frac",
            "sources",
            "off_type_frac",
            "near_enrichment",
            "spray_delta",
            "presence_ratio",
            "pr_score",
            "rank",
        ]
        if c in df.columns
    ]

    print("Top 30 genes by leakage_score:")
    print(summary["top_genes"][top_cols].head(30))
    print()
    print("Marker ranks (GFAP/MBP/PLP1/AQP4 should often rise in CNS datasets):")
    print(summary["marker_summary"])
    print()
    print("Housekeeping-like genes in top 30 (MT-/RPL/RPS/EEF1):")
    print(summary["housekeeping_top"])
    print()
    print("Genes with presence_ratio in [0.4, 0.6] inside top 30 (should be few):")
    print(summary["mid_presence_top"])


if __name__ == "__main__":
    main()
