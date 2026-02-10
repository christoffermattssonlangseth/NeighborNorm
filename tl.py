"""NeighborNorm tool-style API.

Example
-------
```python
import tl

df = tl.leakage_score(adata, layer="counts")
df.head(30)
```
"""

from celltype_adjusted_stickiness import stickiness
from leakage_score import leakage_sanity_check, leakage_score, prepare_leakage_scatter

__all__ = [
    "stickiness",
    "leakage_score",
    "leakage_sanity_check",
    "prepare_leakage_scatter",
]

