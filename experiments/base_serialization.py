from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments.manifests.base import BaseBenchmarkRecord


class SerializedBaseInstanceStore:
    OBJECTS_DIRNAME = "objects"
    OBJECT_FORMAT = "numpy-object-array"

    @classmethod
    def base_dir(cls, record: BaseBenchmarkRecord) -> Path | None:
        if record.manifest_path is None:
            return None
        return Path(record.manifest_path).resolve().parent

    @classmethod
    def objects_dir(cls, record: BaseBenchmarkRecord) -> Path | None:
        base_dir = cls.base_dir(record)
        if base_dir is None:
            return None
        return base_dir / cls.OBJECTS_DIRNAME

    @classmethod
    def object_path(cls, record: BaseBenchmarkRecord) -> Path | None:
        objects_dir = cls.objects_dir(record)
        if objects_dir is None:
            return None
        return objects_dir / f"{record.instance_id}.npy"

    @classmethod
    def metadata_path(cls, record: BaseBenchmarkRecord) -> Path | None:
        objects_dir = cls.objects_dir(record)
        if objects_dir is None:
            return None
        return objects_dir / f"{record.instance_id}.yaml"

    @classmethod
    def _metadata_payload(cls, record: BaseBenchmarkRecord) -> Dict[str, Any]:
        return {
            "object_format": cls.OBJECT_FORMAT,
            "object_file": f"{record.instance_id}.npy",
            "record": record.to_dict(),
        }

    @classmethod
    def _metadata_matches_record(cls, record: BaseBenchmarkRecord, payload: Dict[str, Any]) -> bool:
        return (
            str(payload.get("object_format", "")) == cls.OBJECT_FORMAT
            and payload.get("record") == record.to_dict()
        )

    @classmethod
    def load_instance(cls, record: BaseBenchmarkRecord):
        metadata_path = cls.metadata_path(record)
        object_path = cls.object_path(record)
        if metadata_path is None or object_path is None:
            return None
        if not metadata_path.exists() or not object_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(metadata, dict) or not cls._metadata_matches_record(record, metadata):
            return None
        if str(metadata.get("object_file", "")) != object_path.name:
            return None
        try:
            values = np.load(object_path, allow_pickle=True)
        except (OSError, ValueError, TypeError, AttributeError, ModuleNotFoundError):
            return None
        flat_values = np.asarray(values, dtype=object).reshape(-1)
        if len(flat_values) != 1:
            return None
        return flat_values[0]

    @classmethod
    def write_instance(cls, record: BaseBenchmarkRecord, instance) -> None:
        metadata_path = cls.metadata_path(record)
        object_path = cls.object_path(record)
        if metadata_path is None or object_path is None:
            return
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_tmp = object_path.with_suffix(object_path.suffix + ".tmp")
        metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        try:
            with object_tmp.open("wb") as object_file:
                np.save(object_file, np.array([instance], dtype=object), allow_pickle=True)
            metadata_tmp.write_text(json.dumps(cls._metadata_payload(record), indent=2, sort_keys=True))
            object_tmp.replace(object_path)
            metadata_tmp.replace(metadata_path)
        finally:
            object_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
