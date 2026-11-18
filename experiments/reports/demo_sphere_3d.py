"""Figure 19: top-down time windows of the saved 32-UAV sphere exchange."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.reports.style import configure_matplotlib_for_latex

configure_matplotlib_for_latex(backend='Agg')

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from experiments.reports.common import _save_figure, plt


class SphereExchangeFigure:
    SOURCE_ROOT = Path('data/solutions')
    CASE = 'demo_r32_sphere_exchange'
    ZOOMS = (1, 4, 5, 3, 1)
    WINDOW_STARTS = (0., 1., 1.4, 1.9, 2.4)

    @staticmethod
    def position(segments, time):
        for segment in segments:
            if time <= segment['t_end'] + 1e-10:
                dt = segment['t_end'] - segment['t_start']
                weight = 0. if dt <= 0 else np.clip((time-segment['t_start'])/dt, 0., 1.)
                return (1-weight)*np.asarray(segment['start']) + weight*np.asarray(segment['goal'])
        return np.asarray(segments[-1]['goal'])

    @classmethod
    def draw(cls, source, prefix):
        audit = json.loads((source/cls.CASE/'trajopt_status.json').read_text())
        checked, = [c for c in audit['cases'] if c['name']==cls.CASE]
        if checked['state'] != 'complete':
            raise ValueError('Figure 19 requires validated MRMP and smoothed trajectories.')
        paths = [source/cls.CASE/f'{stage}_solution.json' for stage in ('mrmp','trajopt')]
        stages = [json.loads(p.read_text()) for p in paths]
        for payload in stages:
            assert payload['is_success'] and len(payload['trajectories']) == 32
            assert payload['window_alpha'] == 2 and payload['epsilon'] == 8 and payload['budget'] == 120
        colors = plt.get_cmap('turbo')(np.linspace(0., 1., 32))
        fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.2))
        windows = []
        for row, payload in enumerate(stages):
            boundaries = [*cls.WINDOW_STARTS, float(payload['makespan'])]
            assert np.all(np.diff(boundaries) > 0)
            windows.append(boundaries)
            for column, (start, end, zoom) in enumerate(zip(boundaries[:-1], boundaries[1:], cls.ZOOMS)):
                ax = axes[row, column]
                extent = 1./zoom
                ax.add_patch(Circle((0., 0.), extent, fill=False,
                                    edgecolor='#cccccc' if zoom==1 else '#999999',
                                    lw=.8, ls='-' if zoom==1 else ':', zorder=0))
                for color, trajectory in zip(colors, payload['trajectories'], strict=True):
                    segments = trajectory['segments']
                    lines, line_colors = [], []
                    for segment in segments:
                        left, right = max(start, segment['t_start']), min(end, segment['t_end'])
                        if right <= left:
                            continue
                        # Subdivision makes the time fade independent of optimizer sampling.
                        times = np.linspace(left, right, max(2, int(np.ceil((right-left)/.025))+1))
                        points = [cls.position([segment], time)[:2] for time in times]
                        for index in range(len(points)-1):
                            lines.append([points[index], points[index+1]])
                            rgba = color.copy()
                            rgba[3] = .12 + .82*(.5*(times[index]+times[index+1])-start)/(end-start)
                            line_colors.append(rgba)
                    ax.add_collection(LineCollection(lines, colors=line_colors, linewidths=1.05, zorder=2))
                    begin, goal = np.asarray(segments[0]['start'])[:2], np.asarray(segments[-1]['goal'])[:2]
                    point = cls.position(segments, end)[:2]
                    ax.scatter(*begin, s=24, marker='o', facecolors='none', edgecolors=[color], linewidths=.6, zorder=3)
                    ax.scatter(*goal, s=26, marker='s', facecolors='none', edgecolors=[color], linewidths=.6, zorder=3)
                    ax.scatter(*point, s=21, c=[color], edgecolors='white', linewidths=.45, zorder=4)
                ax.set(xlim=(-1.04*extent, 1.04*extent), ylim=(-1.04*extent, 1.04*extent), aspect='equal')
                ax.axis('off')
                ax.text(.99, .99, rf'${zoom}\times$', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9, color='#444444')
                ax.text(.5, -.065, rf'({chr(97+column)}{row+1}) $t\in[{start:.1f},{end:.1f}]$',
                        transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#444444')
        fig.subplots_adjust(left=.003, right=.997, bottom=.065, top=.997, wspace=.095, hspace=.24)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, prefix, ('pdf','png'))
        plt.close(fig)
        report = dict(instance_id=stages[0]['instance_id'], projection='xy', layout=[2,5],
                      zoom_factors=cls.ZOOMS, time_boundaries=windows,
                      mrmp_runtime=stages[0]['runtime'], trajopt_runtime=stages[1]['global_trajopt']['runtime'],
                      makespan=stages[1]['makespan'],
                      sources=[str(p) for p in paths])
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--source-root', type=Path, default=cls.SOURCE_ROOT)
        parser.add_argument('--output-prefix', type=Path, default=Path('latex/figs/demo_sphere_3d_traj'))
        args = parser.parse_args()
        cls.draw(args.source_root, args.output_prefix)


if __name__ == '__main__':
    SphereExchangeFigure.main()
