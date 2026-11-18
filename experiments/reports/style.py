"""Shared matplotlib configuration for paper-style visualizations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TMP_ROOT = Path(tempfile.gettempdir())
_MPLCONFIGDIR = _TMP_ROOT / "stgcs-devel-matplotlib"
_XDG_CACHE_HOME = _TMP_ROOT / "stgcs-devel-cache"

_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))


def _register_latin_modern_fonts() -> None:
    import matplotlib.font_manager as font_manager

    texlive_root = Path("/usr/local/texlive")
    if not texlive_root.exists():
        return

    for pattern in (
        "lmroman10-regular.otf",
        "lmroman10-bold.otf",
        "lmroman10-italic.otf",
        "lmroman10-bolditalic.otf",
        "lmroman12-regular.otf",
    ):
        for font_path in texlive_root.glob(f"*/texmf-dist/fonts/opentype/public/lm/{pattern}"):
            font_manager.fontManager.addfont(str(font_path))


def configure_matplotlib_for_latex(*, backend: str | None = None, font_size: float = 10.0) -> None:
    """Match matplotlib text rendering to the current LaTeX paper style.

    The paper currently uses the default LaTeX Computer Modern family rather
    than the Times option in ``sagej.cls``. We therefore use Latin Modern /
    Computer Modern fonts directly in matplotlib so that both vector and raster
    outputs stay runnable without requiring the external ``dvipng`` tool.
    """

    import matplotlib

    if backend is not None:
        matplotlib.use(backend)

    _register_latin_modern_fonts()

    matplotlib.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Latin Modern Roman",
            "font.serif": [
                "Latin Modern Roman",
                "cmr10",
                "STIX Two Text",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": 0.9 * font_size,
            "ytick.labelsize": 0.9 * font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
