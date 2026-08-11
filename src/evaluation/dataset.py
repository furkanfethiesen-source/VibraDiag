"""
VibraDiag Retrieval Benchmark Dataset Loader
===========================================
Bu dosya benchmark datasetlerini yüklemek ve Pydantic ile doğrulamak için gerekli altyapıyı içerir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field, model_validator


class GoldItemSchema(BaseModel):
    """
    Benchmark datasetinin tekil bir sorusunu temsil eden doğrulanmış Pydantic modeli.
    """
    id: str = Field(alias="_id", default="")
    question: str
    ground_truth_answer: str
    fault_type: str = "unknown"
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_child_chunk_ids: list[str] = Field(default_factory=list)
    expected_parent_chunk_ids: list[str] = Field(default_factory=list)
    expected_visual_chunk_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def populate_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "_id" in data and "id" not in data:
                data["id"] = data["_id"]
            elif "id" in data and "_id" not in data:
                data["_id"] = data["id"]
            
            raw_expected = data.get("expected_chunk_ids", [])
            if not data.get("expected_child_chunk_ids"):
                data["expected_child_chunk_ids"] = [x for x in raw_expected if "child" in x] or raw_expected
            if not data.get("expected_parent_chunk_ids"):
                data["expected_parent_chunk_ids"] = [x for x in raw_expected if "child" not in x]
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "_id": self.id,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
            "fault_type": self.fault_type,
            "expected_chunk_ids": self.expected_chunk_ids,
            "expected_child_chunk_ids": self.expected_child_chunk_ids,
            "expected_parent_chunk_ids": self.expected_parent_chunk_ids,
            "expected_visual_chunk_ids": self.expected_visual_chunk_ids,
        }


class EvaluationDatasetLoader:
    """
    Benchmark datasetini yükleyen ve şema doğrulamasını yapan sınıf.
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def load_items(self) -> list[GoldItemSchema]:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Gold dataset dosyası bulunamadı: {self.filepath}")

        with open(self.filepath, encoding="utf-8") as f:
            raw_data = json.load(f)

        validated_items: list[GoldItemSchema] = []
        for idx, item in enumerate(raw_data):
            try:
                validated_items.append(GoldItemSchema.model_validate(item))
            except Exception as err:
                raise ValueError(f"Gold dataset satır {idx} şema doğrulaması başarısız oldu: {err}") from err

        logger.info(f"{len(validated_items)} adet doğrulanmış gold soru yüklendi: {self.filepath}")
        return validated_items
