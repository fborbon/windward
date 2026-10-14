"""AMK (additive multiplicative kernel) power-curve regression, vendored verbatim from the MIT-
licensed `dswe` package v0.1.3 (TAMU-AML/DSWE-Python, see LICENSE here) — only the module
rename (_AMK_subroutine -> _amk_subroutine) differs.

Vendored because `import dswe` runs a package __init__ that also imports its DNN/GP models, and
with them torch (~200 MB resident): the API only ever uses AMK, which needs just numpy/pandas/
scipy. The full `dswe` stays in requirements.txt for offline/backtest work.
"""
from .amk import AMK

__all__ = ["AMK"]
