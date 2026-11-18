from __future__ import annotations

import bisect
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


class PlotPalette:
    INK = "#222222"
    BLACK = "#000000"
    NEUTRAL_DARK = "#6f6f6f"
    NEUTRAL_MID = "#777777"
    NEUTRAL_LIGHT = "#a7a7a7"
    WHITE = "white"
    BLUE = "blue"

    HEURISTIC_ZERO = NEUTRAL_LIGHT
    HEURISTIC_MOTION_ONLY = "#4f8f5f"
    HEURISTIC_TRIPLET_RELAXATION = "#3f6fb5"
    HEURISTIC_INTERFACE_TO_SET_TABLE = "#c45a45"
    HEURISTIC_MAX = BLACK

    DOMAIN_RANDOM = "#b6462a"
    DOMAIN_MAZE = INK
    DOMAIN_IRIS = "#1d5e8a"

    HEURISTIC_COLORS: Dict[str, str] = {
        "h_zero": HEURISTIC_ZERO,
        "h_mot": HEURISTIC_MOTION_ONLY,
        "h_tri": HEURISTIC_TRIPLET_RELAXATION,
        "h_tab": HEURISTIC_INTERFACE_TO_SET_TABLE,
        "h_max": HEURISTIC_MAX,
    }
    DOMINANCE_COLORS: Dict[str, str] = {
        "GUB": HEURISTIC_ZERO,
        "delta_set": HEURISTIC_MOTION_ONLY,
        "delta_pos": HEURISTIC_TRIPLET_RELAXATION,
        "delta_state": HEURISTIC_INTERFACE_TO_SET_TABLE,
    }
    DOMAIN_COLORS: Dict[str, str] = {
        "grid2d": DOMAIN_RANDOM,
        "rand": DOMAIN_RANDOM,
        "maze": DOMAIN_MAZE,
        "iris2d": DOMAIN_IRIS,
        "iris-2d": DOMAIN_IRIS,
        "iris": DOMAIN_IRIS,
    }
    ST_METHOD_COLORS: Dict[str, str] = {
        "delta-pos": INK,
        "delta-set": NEUTRAL_DARK,
        "delta-set-eps1": "#0072b2",
        "delta-state": "#56b4e9",
        "micp": "#d55e00",
        "micp_rounding": "#e69f00",
        "st_rrt_first": "#009e73",
        "st_rrt_final": "#5abf90",
        "zeta_sipp": "#cc79a7",
        "zeta_sipp_2r": "#9b5f8f",
    }
    MRMP_RULE_COLORS: Dict[str, str] = {
        "lazy": HEURISTIC_MOTION_ONLY,
        "soc": HEURISTIC_TRIPLET_RELAXATION,
        "makespan": HEURISTIC_INTERFACE_TO_SET_TABLE,
        "num_conflicts": HEURISTIC_MAX,
    }
    MRMP_WINDOW_COLORS: Dict[str, str] = {
        "pbs_nc": BLACK,
        "fixed": HEURISTIC_INTERFACE_TO_SET_TABLE,
        "dynamic": HEURISTIC_MOTION_ONLY,
        "dynamic_beta": HEURISTIC_TRIPLET_RELAXATION,
    }

    @classmethod
    def heuristic_color(cls, name: str) -> str:
        return cls.HEURISTIC_COLORS[name]

    @classmethod
    def dominance_color(cls, name: str) -> str:
        return cls.DOMINANCE_COLORS[name]

    @classmethod
    def domain_color(cls, name: str) -> str:
        return cls.DOMAIN_COLORS[name]


def dedupe_result_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    latest_by_key: Dict[tuple[str, float], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["instance_id"]), float(row["budget"]))
        latest_by_key[key] = row
    return list(latest_by_key.values())


class BudgetSuccessCurve:
    @staticmethod
    def thresholds(
        rows_by_planner: Mapping[str, Sequence[Dict[str, object]]],
        planners: Sequence[str],
        field: str,
        max_budget: float | None = None,
    ) -> List[float]:
        success_values: List[float] = []
        observed_values: List[float] = []
        for planner in planners:
            for row in rows_by_planner.get(planner, []):
                value = float(row[field])
                if not math.isfinite(value) or value <= 0.0:
                    continue
                if max_budget is not None:
                    value = min(value, max_budget)
                observed_values.append(value)
                if bool(row["is_success"]):
                    success_values.append(value)

        right_anchor = max_budget if max_budget is not None else (max(observed_values) if observed_values else 1.0)
        left_anchor = max(right_anchor / 100.0, 1e-3)
        if not success_values:
            return [left_anchor, right_anchor]

        thresholds = sorted(set(success_values))
        first_value = thresholds[0]
        left_anchor = max(min(first_value / 2.0, right_anchor / 10.0), 1e-3)
        if left_anchor < first_value:
            thresholds = [left_anchor, *thresholds]
        if not math.isclose(thresholds[-1], right_anchor):
            thresholds.append(right_anchor)
        return thresholds

    @staticmethod
    def success_rates(rows: Sequence[Dict[str, object]], thresholds: Sequence[float], field: str) -> List[float]:
        if not thresholds:
            return []
        if not rows:
            return [math.nan] * len(thresholds)

        success_values = sorted(
            float(row[field])
            for row in rows
            if bool(row["is_success"]) and math.isfinite(float(row[field])) and float(row[field]) > 0.0
        )
        denominator = len(rows)
        return [bisect.bisect_right(success_values, threshold) / denominator for threshold in thresholds]


class ViolinStyle:
    DEFAULT_WIDTH = 0.62

    @staticmethod
    def draw(
        ax: plt.Axes,
        position: float,
        values: Sequence[float],
        color: str,
        width: float = DEFAULT_WIDTH,
        *,
        linewidth: float = 0.8,
        median_linewidth: float = 2.2,
        orientation: str = "horizontal",
    ) -> None:
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"Unsupported violin orientation {orientation!r}.")
        if max(values) != min(values):
            violin = ax.violinplot(
                [values], positions=[position], widths=width,
                orientation=orientation, points=200, bw_method="scott",
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.3)
                body.set_linewidth(linewidth)
        # Keep the median visible above the per-instance points (zorder=3).
        draw_median = ax.hlines if orientation == "vertical" else ax.vlines
        draw_median(np.median(values), position - width * 0.35,
                    position + width * 0.35, color=PlotPalette.BLACK,
                    linewidth=median_linewidth, zorder=4, capstyle="butt",
                    path_effects=[path_effects.Stroke(linewidth=median_linewidth + 1.4,
                                                     foreground=PlotPalette.WHITE),
                                  path_effects.Normal()])


class BoxplotStyle:
    DEFAULT_WIDTH = 0.42
    DEFAULT_FACE_COLOR = "white"
    DEFAULT_WHIS = (5, 95)
    BOX_LINEWIDTH = 1.1
    MEDIAN_LINEWIDTH = 1.25
    WHISKER_LINEWIDTH = 0.9
    SCATTER_WIDTH_FRACTION = 0.55
    SCATTER_SIZE = 8.0
    SCATTER_ALPHA = 0.72

    @classmethod
    def draw(
        cls,
        ax: plt.Axes,
        values: Sequence[Sequence[float]],
        positions: Sequence[float],
        widths: float | Sequence[float],
        color: str,
        facecolor: str = DEFAULT_FACE_COLOR,
        showfliers: bool = False,
        whis: tuple[float, float] | None = DEFAULT_WHIS,
        orientation: str = "vertical",
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "orientation": orientation,
            "positions": positions,
            "widths": widths,
            "manage_ticks": False,
            "patch_artist": True,
            "showfliers": showfliers,
            "boxprops": {
                "facecolor": facecolor,
                "edgecolor": color,
                "linewidth": cls.BOX_LINEWIDTH,
            },
            "medianprops": {
                "color": color,
                "linewidth": cls.MEDIAN_LINEWIDTH,
            },
            "whiskerprops": {
                "color": color,
                "linewidth": cls.WHISKER_LINEWIDTH,
            },
            "capprops": {
                "color": color,
                "linewidth": cls.WHISKER_LINEWIDTH,
            },
        }
        if whis is not None:
            kwargs["whis"] = whis
        return ax.boxplot(values, **kwargs)

    @classmethod
    def scatter_points(
        cls,
        ax: plt.Axes,
        position: float,
        values: Sequence[float],
        width: float = DEFAULT_WIDTH,
        color: str = PlotPalette.INK,
        size: float = SCATTER_SIZE,
        alpha: float = SCATTER_ALPHA,
        orientation: str = "vertical",
    ) -> None:
        jittered_positions = cls.scatter_x_positions(position, width, len(values))
        if orientation == "vertical":
            ax.scatter(
                jittered_positions,
                values,
                s=size,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=alpha,
                zorder=3,
            )
            return
        if orientation == "horizontal":
            ax.scatter(
                values,
                jittered_positions,
                s=size,
                marker="o",
                facecolors=color,
                edgecolors="white",
                linewidths=0.25,
                alpha=alpha,
                zorder=3,
            )
            return
        raise ValueError(f"Unsupported scatter orientation {orientation!r}.")

    @classmethod
    def scatter_x_positions(cls, position: float, width: float, count: int) -> list[float]:
        if count <= 1:
            return [float(position)] * count
        span = float(width) * cls.SCATTER_WIDTH_FRACTION
        return [
            float(position) + span * (idx / (count - 1) - 0.5)
            for idx in range(count)
        ]


def _set_square_cells(axes: np.ndarray) -> None:
    for ax in axes.flat:
        ax.set_box_aspect(1)


def _save_figure(
    fig: plt.Figure,
    output_prefix: Path,
    formats: Sequence[str],
    *,
    bbox_inches: str | None = "tight",
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_prefix.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=240, bbox_inches=bbox_inches)
        print(f"Saved figure to {output_path}")
    plt.close(fig)
