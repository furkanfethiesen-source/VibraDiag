
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict
from dataclasses import dataclass

class _PeakItem(BaseModel):
    order: str
    relative_amplitude: str


class SpectrumSchema(BaseModel):
    content_type: Literal["spectrum"] = "spectrum"
    visual_reasoning: str = Field(
        description="Görseli adım adım incele: Eksen birimlerini, belirgin pik noktalarını ve dalga formu desenlerini açıkla."
    )
    description: str
    domain: Literal["frequency", "time"] = "frequency"
    fault_type: Optional[str] = None
    direction: Optional[str] = None
    phase_criterion: Optional[str] = None
    distinguishing_rule: Optional[str] = None
    peaks: Optional[list[_PeakItem]] = None
    sideband_pattern: Optional[str] = None
    time_domain_pattern: Optional[str] = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class BearingStageSchema(BaseModel):
    content_type: Literal["bearing_stage"] = "bearing_stage"
    visual_reasoning: str = Field(
        description="Görseli adım adım incele: Görünür frekans bölgelerini, ultrasonic/HFD bileşenlerini ve rulman bozulma aşamasını açıkla."
    )
    description: str
    stage: Optional[str] = None
    zones_visible: Optional[list[str]] = None
    hfd_amplitude_trend: Optional[str] = None
    distinguishing_rule: Optional[str] = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class OrbitSchema(BaseModel):
    content_type: Literal["orbit_plot"] = "orbit_plot"
    visual_reasoning: str = Field(
        description="Görseli adım adım incele: Yörünge geometrisini (elips, muz, 8-şekli, iç ilmek vb.) ve olası arıza türünü açıkla."
    )
    description: str
    orbit_shape: Optional[str] = None
    fault_type: Optional[str] = None
    distinguishing_rule: Optional[str] = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class SeverityChartSchema(BaseModel):
    content_type: Literal["severity_chart"] = "severity_chart"
    visual_reasoning: str = Field(
        description="Görseli adım adım incele: Standart normları (ISO vb.), makine gruplarını ve sınır titreşim seviyelerini açıkla."
    )
    description: str
    standard: Optional[str] = None
    groups: Optional[list[str]] = None
    zone_meanings: Optional[dict[str, str]] = None
    measurement_unit: Optional[str] = None
    frequency_range: Optional[str] = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class UnifiedSchema(BaseModel):
    """
    UNIFIED_PROMPT için: caption heuristic'i içeriği önceden
    sınıflandıramadığında kullanılır, bu yüzden content_type önceden
    bilinmiyor. Tüm olası alanları opsiyonel superset olarak taşır.
    """
    content_type: Literal[
        "spectrum", "severity_chart", "bearing_stage", "orbit_plot",
        "diagram", "waveform", "equipment_photo", "decorative",
        "schematic_generic", "logo", "unknown",
    ]
    visual_reasoning: str = Field(
        description="Görseli adım adım incele: Görsel tipini, belirgin desenleri ve teknik özellikleri açıkla."
    )
    description: str
    domain: Optional[Literal["frequency", "time"]] = None
    fault_type: Optional[str] = None
    direction: Optional[str] = None
    phase_criterion: Optional[str] = None
    distinguishing_rule: Optional[str] = None
    peaks: Optional[list[_PeakItem]] = None
    sideband_pattern: Optional[str] = None
    time_domain_pattern: Optional[str] = None
    stage: Optional[str] = None
    zones_visible: Optional[list[str]] = None
    hfd_amplitude_trend: Optional[str] = None
    orbit_shape: Optional[str] = None
    standard: Optional[str] = None
    groups: Optional[list[str]] = None
    zone_meanings: Optional[dict[str, str]] = None
    measurement_unit: Optional[str] = None
    frequency_range: Optional[str] = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"



@dataclass
class TextChunk:
    chunk_id: str
    content_type: str = "text"
    chunk_level: str = "child"          
    parent_id: Optional[str] = None
    page_number: int = 0
    page_range: Optional[list] = None
    section_path: str = "unknown"
    text: str = ""
    token_count: int = 0

@dataclass
class TableChunk:
    chunk_id: str
    content_type: str = "table"
    page_number: int = 0
    section_path: str = "unknown"
    table_title: str = ""
    headers: Optional[list] = None
    rows: Optional[list] = None
    markdown_repr: str = ""

@dataclass
class VisualChunk:
    chunk_id: str
    content_type: str
    page_number: int
    section_path: str
    source_caption: str
    generated_text: str
    structured_data: Optional[dict] = None
    confidence: str = "unknown"      
    needs_review: bool = False       

    def to_metadata(self) -> dict:
        """
        ChromaDB metadata olarak serileştirir (flat dict, nested dict yok).

        structured_data içindeki önemli alanlar (fault_type, domain, direction)
        flat olarak çıkarılır — ChromaDB nested dict desteklemez.
        """
        meta: dict = {
            "chunk_id": self.chunk_id,
            "content_type": self.content_type,
            "page_number": self.page_number,
            "section_path": self.section_path,
            "source_caption": self.source_caption,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }
        if self.structured_data:
            for key in ("fault_type", "domain", "direction", "distinguishing_rule"):
                value = self.structured_data.get(key)
                if value is not None:
                    meta[key] = value
        return meta



