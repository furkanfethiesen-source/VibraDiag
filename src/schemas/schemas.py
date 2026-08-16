
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
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

class ClassifierResult(BaseModel):
    """Lojistik regresyon complexity classifier çıktısı."""
    is_complex: bool
    confidence: float  


class DecomposeResult(BaseModel):
    """LLM decomposer çıktısı."""
    sub_queries: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class TokenUsageInfo(BaseModel):
    """Token kullanımı ve tahmini maliyet modeli."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class CheckerResult(BaseModel):
    """Tekil bir doğrulama (checker) adımının sonucu."""
    checker_name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="0.0 - 1.0 arası güven/uyum skoru")
    rationale: str = Field(default="", description="Kararın gerekçesi veya hata açıklaması")
    token_usage: TokenUsageInfo = Field(default_factory=TokenUsageInfo)
    latency_ms: float = 0.0
    execution_error: bool = False
    error_message: Optional[str] = None



class CorrectionDecision(BaseModel):
    """Self-Corrector node'unun nihai yönlendirme kararı."""
    status: Literal["approved", "retry_retrieval", "retry_generation", "flagged"]
    trigger_reasons: list[str] = Field(default_factory=list)
    checker_results: dict[str, CheckerResult] = Field(default_factory=dict)
    reformulated_query: Optional[str] = None
    strategy_used: Optional[str] = None
    total_tokens: TokenUsageInfo = Field(default_factory=TokenUsageInfo)
    total_latency_ms: float = 0.0
    needs_review: bool = False
    warning_message: Optional[str] = None


class CorrectorBenchmarkItem(BaseModel):
    """Sentetik Corrector değerlendirme veri seti öğesi."""
    id: str
    scenario_type: Literal["clean", "dsp_mismatch", "unfaithful", "sub_query_gap"]
    user_query: str
    dsp_ground_truth: Optional[dict[str, Any]] = None
    retrieved_contexts: list[dict[str, Any]] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    candidate_response: str
    expected_decision: Literal["approved", "retry_retrieval", "retry_generation", "flagged"]
    expected_failing_checkers: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

