"""Plot Fig. 9 from the saved spatial GCS instances used by the ST dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend="Agg")

from experiments.base import BaseInstanceFactory
from experiments.manifests.base import load_manifest
from experiments.reports.common import _save_figure, plt


class SpatialGCSDomainFigure:
    # IRIS seed 74 has separated static obstacles spread across all four quadrants.
    EXAMPLES = (("grid2d", "rand", 1), ("maze", "maze", 0), ("iris-2d", "iris", 74))

    @classmethod
    def render(cls, base_root: Path, output_prefix: Path, formats: list[str]) -> None:
        examples = []
        for domain, label, seed in cls.EXAMPLES:
            record, = (r for r in load_manifest(base_root / domain / "manifest.json")
                       if r.spatial_seed == seed)
            instance = BaseInstanceFactory.load_cached_instance(record)
            if instance is None:
                raise ValueError(f"Missing or mismatched saved instance: {record.instance_id}")
            env = instance.env
            graph = instance.stgcs.G
            if (env.dim != 2 or env.O_Dynamic
                    or len(env.C_Space) != record.stgcs_num_vertices
                    or graph.number_of_nodes() != record.stgcs_num_vertices
                    or graph.number_of_edges() != record.stgcs_num_edges):
                raise ValueError(f"Spatial GCS does not match its manifest: {record.instance_id}")
            examples.append((label, env))
            print(f"{label}: {record.instance_id}; {len(env.C_Space)} regions, "
                  f"{graph.number_of_edges()} directed edges, {len(env.O_Static)} static obstacles")

        lower = np.min([env.lb for _, env in examples], axis=0)
        upper = np.max([env.ub for _, env in examples], axis=0)
        center = (lower + upper) / 2
        half_width = 0.53 * float(np.max(upper - lower))
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        palette = plt.get_cmap("Set3")
        for ax, (label, env) in zip(axes, examples, strict=True):
            for index, vertices in enumerate(env.C_Space):
                ax.fill(vertices[:, 0], vertices[:, 1],
                        facecolor=palette(index % palette.N), edgecolor="#444444",
                        linewidth=0.6, alpha=0.65)
            for obstacle in env.O_Static:
                vertices = obstacle.vertices
                ax.fill(vertices[:, 0], vertices[:, 1],
                        facecolor="#222222", edgecolor="black", linewidth=0.4, zorder=3)
            ax.set(xlim=(center[0] - half_width, center[0] + half_width),
                   ylim=(center[1] - half_width, center[1] + half_width), aspect="equal")
            ax.set_title(label, fontsize=16, pad=8)
            ax.axis("off")
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.86, wspace=0.06)
        _save_figure(fig, output_prefix, formats)

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--base-root", type=Path, default=Path("data/instances/st_planning/base"))
        parser.add_argument("--output-prefix", type=Path, default=Path("latex/figs/exp_domains"))
        parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
        args = parser.parse_args()
        cls.render(args.base_root, args.output_prefix, args.formats)


if __name__ == "__main__":
    SpatialGCSDomainFigure.main()
