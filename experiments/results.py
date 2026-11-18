from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Sequence

from experiments.manifests.st_planning import STHeuristicAblationRecord, STHeuristicAblationResultEntry


class STResultStore:
    RESULT_FIELDS = (
        "name",
        "is_success",
        "runtime",
        "cost",
        "num_expanded_nodes",
        "num_generated_nodes",
        "instance_id",
        "budget",
        "group",
        "source_domain",
        "base_instance_id",
        "obstacle_count",
        "stgcs_num_vertices",
        "stgcs_num_edges",
    )

    @classmethod
    def result_key(cls, instance_id: str, budget: float) -> tuple[str, float]:
        return str(instance_id), float(budget)

    @staticmethod
    def planner_output_path(output_root: str | Path, planner_name: str) -> Path:
        target_dir = Path(output_root)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{planner_name.replace('/', '_')}.csv"

    @classmethod
    def load_result_rows(cls, path: str | Path) -> list[Dict[str, object]]:
        target = Path(path)
        if not target.exists():
            return []
        rows: list[Dict[str, object]] = []
        with target.open(newline="") as fh:
            reader = csv.reader(fh)
            expected_width = len(cls.RESULT_FIELDS)
            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != expected_width:
                    raise ValueError(
                        f"{target}:{line_number}: expected {expected_width} ST result fields, got {len(row)}."
                    )
                rows.append(
                    {
                        "name": row[0],
                        "is_success": row[1].lower() == "true",
                        "runtime": float(row[2]) if row[2] != "inf" else math.inf,
                        "cost": float(row[3]) if row[3] != "inf" else math.inf,
                        "num_expanded_nodes": int(float(row[4])),
                        "num_generated_nodes": int(float(row[5])),
                        "instance_id": row[6],
                        "budget": float(row[7]),
                        "group": row[8],
                        "source_domain": row[9],
                        "base_instance_id": row[10],
                        "obstacle_count": int(float(row[11])),
                        "stgcs_num_vertices": int(float(row[12])),
                        "stgcs_num_edges": int(float(row[13])),
                    }
                )
        return rows

    @classmethod
    def completed_result_keys(cls, output_path: Path) -> set[tuple[str, float]]:
        return {
            cls.result_key(str(row["instance_id"]), float(row["budget"]))
            for row in cls.load_result_rows(output_path)
        }

    @classmethod
    def append_result_row(
        cls,
        output_path: str | Path,
        planner_name: str,
        record: STHeuristicAblationRecord,
        budget: float,
        entry: STHeuristicAblationResultEntry,
    ) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", newline="") as fh:
            writer = csv.writer(fh)
            cls.write_result_row(writer, planner_name, record, budget, entry)

    @classmethod
    def write_result_rows(
        cls,
        output_path: str | Path,
        planner_name: str,
        rows: Sequence[tuple[STHeuristicAblationRecord, STHeuristicAblationResultEntry]],
        budget: float,
    ) -> None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as fh:
            writer = csv.writer(fh)
            for record, entry in rows:
                cls.write_result_row(writer, planner_name, record, budget, entry)

    @staticmethod
    def write_result_row(
        writer,
        planner_name: str,
        record: STHeuristicAblationRecord,
        budget: float,
        entry: STHeuristicAblationResultEntry,
    ) -> None:
        writer.writerow(
            [
                planner_name,
                entry.is_success,
                entry.runtime,
                entry.cost,
                entry.num_expanded_nodes,
                entry.num_generated_nodes,
                record.instance_id,
                float(budget),
                record.group,
                record.source_domain,
                record.base_instance_id,
                len(record.dynamic_obstacles),
                record.stgcs_num_vertices,
                record.stgcs_num_edges,
            ]
        )
