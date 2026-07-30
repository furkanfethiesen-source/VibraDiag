
"""
Vision LLM Destekli PDF Ingestion Pipeline
===================================================================

Amaç: Vibration Analysis Guide gibi teknik dökümanlardaki figürleri
(FFT spektrumları, severity chart'lar, bearing stage diyagramları,
orbit plotları vb.) Gemini vision LLM kullanarak yapılandırılmış
metne çevirip chunk (VisualChunk) üretmek.

NOT: Bu modül SADECE chunk üretir. Embedding/ChromaDB'ye yazma işi
başka bir modülün sorumluluğu.

Mimari notlar:
- Allow-list: Bu dokümanda 110 görselden ~25'i teşhis açısından
    işlevsel olduğu için belirlendi. Vision LLM çağrısından ÖNCE
    filtrelenir — hem maliyeti düşürür hem alakasız görselleri
    (equipment photo, dekoratif şematik) baştan eler.
- content_type'a özel prompt/şema: spectrum, bearing_stage, orbit_plot
    ve severity_chart farklı yapısal alanlara ihtiyaç duyuyor.
- domain routing: "spectrum" içinde bile frekans-domeni (peaks) ve
    zaman-domeni (time_domain_pattern) görseller farklı şema gerektiriyor
    (örn. Fig 5.12 cracked tooth zaman-domeni dalga formu).
- Çapraz doğrulama: LLM'in bulduğu fault_type, sayfadaki "Figure X.Y..."
    caption metniyle kabaca karşılaştırılır; uyuşmazsa needs_review=True.
- Hata izolasyonu: Bir görselin vision LLM çağrısı başarısız olursa
    tüm batch durmaz, o görsel loglanıp atlanır.

Gereksinimler:
    pip install pymupdf google-genai chromadb --break-system-packages

Ortam değişkeni:
    export GEMINI_API_KEY="..."
"""

import os
import io
import json
import re
import time
import base64
from dataclasses import dataclass, asdict, field
from typing import Optional
from ingestion.section_utils import (
    _extract_section_headings,
    _resolve_section_path
)
from config_loader import load_appcfg, load_prompts_cfg
from schemas.schemas import (
    SpectrumSchema,
    BearingStageSchema,
    OrbitSchema,
    SeverityChartSchema,
    UnifiedSchema,
    VisualChunk
)
import fitz  
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

app_cfg = load_appcfg()
prompts_cfg = load_prompts_cfg()


class VisionLLMError(RuntimeError):
    """
    Vision LLM çağrısı kalıcı olarak başarısız olduğunda fırlatılır.
    raw_text: modelin ürettiği (varsa) ham metin -- post-mortem debug
    ve failed_figures logu için saklanır.
    """
    def __init__(self, message: str, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text



_ORBIT_SHAPE_FAULT_MAP: dict[str, set[str]] = {
    "inner_loop": {"rub", "hit_and_bounce", "rubbing"},
    "circular": {"none", "normal", "healthy", "unbalance"},
    "elliptical": {"unbalance", "misalignment", "resonance", "bearing_wear"},
    "banana_shape": {"misalignment"},
    "figure_eight": {"misalignment"},
    "multiple_loops": {"looseness", "mechanical_looseness"},
}

_DIAGRAM_LEAK_KEYWORDS = {
    "1st cycle", "2nd cycle", "1 rev", "rigid shaft", "flexible shaft",
    "schematic", "schematic diagram", "pinned at this point",
}


INGEST_ALLOWLIST: set[tuple[int, Optional[int]]] = {
    (30, None), (32, None), (33, None), (35, None), (36, None),
    (37, None), (38, None), (39, None),
    (42, None), (43, None), (44, None), (45, None), (46, None),
    (51, None), (52, None), (53, None),
    (54, None), (55, None), (56, None), (57, None),
    (59, None), (60, None),
}


def is_page_allowlisted(page_number: int, local_index: int) -> bool:
    """Görsel allow-list'te mi kontrol eder."""
    if (page_number, None) in INGEST_ALLOWLIST:
        return True
    return (page_number, local_index) in INGEST_ALLOWLIST


def extract_figure_images(pdf_path: str | None = None, *, doc: fitz.Document | None = None) -> list[dict]:
    """
    Her sayfadaki gömülü görselleri (figürleri) PNG bytes olarak çıkarır.
    Not: Bu basit sürüm sayfadaki tüm raster görselleri alır; asıl
    filtreleme allow-list ile pipeline seviyesinde yapılır.

    Args:
        pdf_path: PDF dosya yolu. doc verilmediyse bu kullanılır.
        doc: Önceden açılmış fitz.Document nesnesi. Verilirse pdf_path
            yok sayılır ve doc kapatılmaz (çağıran sorumludur).
    """
    owns_doc = False
    if doc is None:
        if pdf_path is None:
            raise ValueError("pdf_path veya doc parametrelerinden biri gerekli")
        doc = fitz.open(pdf_path)
        owns_doc = True

    figures = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        seen_xrefs = set()
        dedup_index = 0
        for img in image_list:
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            figures.append({
                "page_number": page_index + 1,
                "image_bytes": image_bytes,
                "image_ext": base_image["ext"],
                "local_index": dedup_index,
                "xref": xref
            })
            dedup_index += 1

    if owns_doc:
        doc.close()
    return figures


_parsing_prompts = prompts_cfg.parsing_prompts

SPECTRUM_PROMPT = _parsing_prompts.get("spectrum_prompt", "")
SEVERITY_CHART_PROMPT = _parsing_prompts.get("severity_chart_prompt", "")
BEARING_STAGE_PROMPT = _parsing_prompts.get("bearing_stage_prompt", "")
ORBIT_PROMPT = _parsing_prompts.get("orbit_prompt", "")
UNIFIED_PROMPT = _parsing_prompts.get("unified_prompt", "")

_SEVERITY_KEYWORDS = {"severity", "iso 10816", "iso10816", "vdi 2056",
                    "vdi2056", "ird mechanalysis"}
_BEARING_STAGE_KEYWORDS = {"bearing defect", "stage 1", "stage 2", "stage 3",
                        "stage 4", "hfd", "zone a", "zone b", "zone c",
                        "zone d"}
_ORBIT_KEYWORDS = {"orbit", "lissajous"}
_SPECTRUM_KEYWORDS = {"fft", "spectrum", "waveform", "bode",
                    "nyquist", "cepstrum", "waterfall", "spectrogram",
                    "envelope", "phase relationship", "phase relation"}

_SKIP_CAPTION_KEYWORDS = {"equipment", "photograph", "courtesy", "logo",
                        "photo of", "installed", "mounting", "accessories"}


def _pre_flight_skip(caption: str, image_bytes: bytes) -> bool:
    """
    LLM çağrısından ÖNCE görselin skip edilip edilemeyeceğini belirler.
    """
    if not caption:
        return False
    caption_lower = caption.lower()
    if any(kw in caption_lower for kw in _SKIP_CAPTION_KEYWORDS):
        return True
    if len(image_bytes) < 5_000:
        return True
    return False


def _extract_page_caption(page: fitz.Page) -> str:
    """Sayfadaki 'Figure X.Y ...' caption metinlerini birleştirip döner."""
    text = page.get_text()
    captions = re.findall(r'Figure\s+\d+\.\d+[^\n]+', text)
    return " ".join(captions).strip() if captions else ""


def _prompt_meta() -> dict[int, tuple[str, type[BaseModel]]]:
    """
    Prompt string'inin id()'sine göre (label, response_schema) döner.
    Fonksiyon olarak tutulmasının sebebi: modül import sırasında
    prompts_cfg henüz farklı bir değerle yeniden set edilmiş olabilir
    (test/reload senaryoları); her çağrıda güncel referanslarla eşleşir.
    """
    return {
        id(SEVERITY_CHART_PROMPT): ("SEVERITY", SeverityChartSchema),
        id(BEARING_STAGE_PROMPT): ("BEARING_STAGE", BearingStageSchema),
        id(ORBIT_PROMPT): ("ORBIT", OrbitSchema),
        id(SPECTRUM_PROMPT): ("SPECTRUM", SpectrumSchema),
        id(UNIFIED_PROMPT): ("UNIFIED", UnifiedSchema),
    }


def prompt_label_and_schema(prompt: str) -> tuple[str, type[BaseModel]]:
    """Verilen prompt string'i için (etiket, response_schema) döner."""
    return _prompt_meta().get(id(prompt), ("UNIFIED", UnifiedSchema))


def classify_and_select_prompt(page: fitz.Page) -> tuple[str, str]:
    """
    Caption-based heuristic ile görselin tipini tahmin eder ve uygun
    prompt'u seçer. Caption metnini de döndürür — böylece çağıran
    taraf _extract_page_caption'ı tekrar çağırmak zorunda kalmaz.

    Öncelik sırası önemli: bearing_stage ve orbit caption'ları çoğu
    zaman "fft"/"spectrum" kelimesini de içerebildiği için, genel
    spectrum kontrolünden ÖNCE kontrol edilmeli.

    Returns:
        (prompt, caption_text) tuple'ı
    """
    caption = _extract_page_caption(page)
    caption_lower = caption.lower()

    if any(kw in caption_lower for kw in _SEVERITY_KEYWORDS):
        return SEVERITY_CHART_PROMPT, caption
    if any(kw in caption_lower for kw in _BEARING_STAGE_KEYWORDS):
        return BEARING_STAGE_PROMPT, caption
    if any(kw in caption_lower for kw in _ORBIT_KEYWORDS):
        return ORBIT_PROMPT, caption
    if any(kw in caption_lower for kw in _SPECTRUM_KEYWORDS):
        return SPECTRUM_PROMPT, caption

    print(f"  [uyarı] Caption keyword eşleşmesi yok, UNIFIED_PROMPT kullanılıyor: "
        f"\"{caption[:80]}\"")
    return UNIFIED_PROMPT, caption


def _sanitize_and_parse_json(raw_text: str) -> dict:
    """
    Model çıktısını JSON'a çevirir. İki yaygın kırılma modunu tolere eder:
    1) Model response_mime_type=json'a rağmen çıktıyı ```json ... ```
        fence'i içine sarmışsa, fence temizlenir.
    2) Model geçerli bir JSON objesinin ARDINDAN ekstra metin/ikinci bir
        obje eklemişse ("Extra data" hatası), json.loads yerine
        JSONDecoder.raw_decode kullanılır -- bu, ilk geçerli JSON değerini
        alır ve sonrasını yok sayar (json.loads sert biçimde tüm metnin
        tek bir JSON olmasını ister, raw_decode istemez).
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def caption_figure_with_vision_llm(
    client: genai.Client,
    image_bytes: bytes,
    prompt: str | None = None,
    mime_type: str = "image/png",
    max_retries: int | None = None,
    response_schema: type[BaseModel] | None = None,
) -> dict:
    """
    Gemini vision modeline görseli ve seçilen prompt'u gönderip
    yapılandırılmış JSON caption alır.

    Retry politikası:
    - 429/503, timeout, bağlantı hataları → exponential backoff ile retry
    - MAX_TOKENS ile kesilen yanıt → retryable (kesik JSON zaten
        parse edilemez; tekrar denemek genelde farklı/tam bir çıktı üretir)
    - JSON parse hatası (fence/extra-data temizliğine rağmen) → artık
        KALICI SAYILMAZ; max_retries içinde tekrar denenir, son denemede
        de başarısız olursa VisionLLMError (ham metinle birlikte) fırlatılır
    - Diğer hatalar → hata mesajı analiz edilerek karar verilir

    response_schema verilirse (bkz. SpectrumSchema, BearingStageSchema,
    OrbitSchema, SeverityChartSchema, UnifiedSchema) Gemini'nin controlled
    generation / structured output modu devreye girer: model, şema dışı
    sözdizimsel olarak bozuk JSON üretemez hale gelir. Bu, visual_chunks_
    failed.json'daki "Expecting ',' delimiter" / "Extra data" hatalarının
    kök nedenini (şemasız serbest-metin JSON üretimi) ortadan kaldırır.
    """
    if prompt is None:
        prompt = SPECTRUM_PROMPT
    if max_retries is None:
        max_retries = app_cfg.vision_llm.get("max_retries", 4)

    model_name = app_cfg.vision_llm.get("model", "gemini-3.5-flash")
    max_output_tokens = app_cfg.vision_llm.get("max_output_tokens", 2048)

    _retryable_keywords = {"429", "rate limit", "resource exhausted",
                        "503", "service unavailable", "deadline exceeded",
                        "timeout", "connection"}

    last_raw_text: str | None = None

    for attempt in range(max_retries):
        try:
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=max_output_tokens,
                temperature=0,
            )
            if response_schema is not None:
                gen_config.response_schema = response_schema

            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                ),
                config=gen_config,
            )
            if response.text is None:
                raise ValueError("Vision LLM yanıtı boş (response.text is None)")

            finish_reason = None
            if response.candidates:
                finish_reason = str(getattr(response.candidates[0], "finish_reason", "")).upper()
            if finish_reason and "MAX_TOKENS" in finish_reason:
                last_raw_text = response.text
                wait = 2 ** attempt
                print(f"  [retry {attempt+1}/{max_retries}] yanıt MAX_TOKENS nedeniyle "
                    f"kesildi -> {wait}s bekleyip tekrar deneniyor")
                time.sleep(wait)
                continue

            last_raw_text = response.text
            return _sanitize_and_parse_json(response.text)

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [json-repair retry {attempt+1}/{max_retries}] "
                    f"geçersiz JSON: {e} -> {wait}s bekleyip tekrar deneniyor")
                time.sleep(wait)
                continue
            raise VisionLLMError(
                f"Vision LLM kalıcı hata (JSON, {max_retries} deneme tükendi): {e}",
                raw_text=last_raw_text,
            ) from e

        except Exception as e:
            error_str = str(e).lower()
            if not any(kw in error_str for kw in _retryable_keywords):
                # Tanımlanamayan kalıcı hata
                raise VisionLLMError(
                    f"Vision LLM kalıcı hata (retry yapılmadı): {e}",
                    raw_text=last_raw_text,
                ) from e

            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] geçici hata: {e} -> {wait}s bekleniyor")
            time.sleep(wait)

    raise VisionLLMError(
        "Vision LLM çağrısı max_retries sonunda başarısız oldu",
        raw_text=last_raw_text,
    )


INGESTIBLE_TYPES = {"spectrum", "severity_chart", "bearing_stage",
                    "orbit_plot", "diagram"}

SKIP_TYPES = {"equipment_photo", "decorative", "schematic_generic", "logo"}


def is_worth_ingesting(caption_json: dict) -> bool:
    """
    content_type whitelist'e göre görselin ingest'e değer olup
    olmadığını belirler. Bilinmeyen tipler için description
    uzunluğuna dayalı fallback heuristic uygulanır.
    """
    content_type = caption_json.get("content_type", "unknown")

    if content_type in INGESTIBLE_TYPES:
        return True
    if content_type in SKIP_TYPES:
        return False

    desc = caption_json.get("description", "")
    return len(desc) > 50


def _cross_validate_fault_type(fault_type: str | None, source_caption: str) -> bool:
    """
    LLM'in bulduğu fault_type ile sayfadaki 'Figure X.Y ...' metnini
    kabaca karşılaştırır. Uyuşmazsa needs_review tetiklenir.
    Karşılaştıracak veri yoksa güvenli varsayılır (review'a düşürmez) —
    amaç yanlış-pozitif review yükü yaratmamak.
    """
    if not fault_type or not source_caption:
        return True

    normalized = fault_type.lower().replace("_", " ")
    caption_lower = source_caption.lower()
    key_terms = [t for t in normalized.split() if len(t) > 3]
    if not key_terms:
        return True
    return any(term in caption_lower for term in key_terms)


def _orbit_consistency_flag(orbit_shape: str | None, fault_type: str | None) -> bool:
    """
    orbit_shape ile fault_type kaynak dokümandaki sabit eşleşmeye uyuyor mu?
    Uymuyorsa True (review gerekli) döner. Bilgi eksikse konservatif
    davranıp review'a düşürmez (yanlış-pozitif review yükü yaratmamak için).

    Örnek: "inner_loop" -> yalnızca hit-and-bounce/rub anlamına gelir;
    "misalignment" DEĞİL (misalignment eliptik/muz/8-şekli üretir). Bu
    tablo olmadan fault_type serbest metinden geldiği için model bunu
    karıştırabiliyordu (bkz. fig_p60_vision_0).
    """
    if not orbit_shape or not fault_type:
        return False
    allowed = _ORBIT_SHAPE_FAULT_MAP.get(orbit_shape.lower().replace(" ", "_"))
    if allowed is None:
        return False
    normalized_fault = fault_type.lower().replace(" ", "_")
    return normalized_fault not in allowed


def _spectrum_diagram_leak_flag(generated_text: str) -> bool:
    """
    content_type="spectrum" olarak etiketlenmiş ama generated_text aslında
    bir mekanik şema/diyagramı tarif ediyorsa True döner (bkz. fig_p35_
    vision_1: "1st CYCLE / 2nd CYCLE / 1 REV" tarifi ile birlikte uydurma
    peak verisi üretilmişti).
    """
    text_lower = generated_text.lower()
    return any(kw in text_lower for kw in _DIAGRAM_LEAK_KEYWORDS)


def build_chunk(page_number: int, local_index: int, caption_json: dict,
                source_caption: str = "",
                section_path: str = "unknown",
                prefix: str = "img") -> VisualChunk:
    """
    LLM JSON yanıtından VisualChunk oluşturur. content_type'a göre
    farklı structured_data şeması doldurulur:
    - spectrum: domain'e göre frekans (peaks) veya zaman
        (time_domain_pattern) alt-şeması
    - bearing_stage: rulman bozulma aşaması bilgisi
    - orbit_plot: orbit şekli bilgisi
    - severity_chart: standart/grup/zon bilgisi
    """
    content_type = caption_json.get("content_type", "spectrum")

    structured: dict = {}

    if content_type == "spectrum":
        domain = caption_json.get("domain", "frequency")
        structured = {
            "fault_type": caption_json.get("fault_type"),
            "domain": domain,
            "direction": caption_json.get("direction"),
            "phase_criterion": caption_json.get("phase_criterion"),
            "distinguishing_rule": caption_json.get("distinguishing_rule"),
        }
        if domain == "time":
            structured["time_domain_pattern"] = caption_json.get("time_domain_pattern")
        else:
            structured["peaks"] = caption_json.get("peaks")
            structured["sideband_pattern"] = caption_json.get("sideband_pattern")

    elif content_type == "bearing_stage":
        structured = {
            "stage": caption_json.get("stage"),
            "zones_visible": caption_json.get("zones_visible"),
            "hfd_amplitude_trend": caption_json.get("hfd_amplitude_trend"),
            "distinguishing_rule": caption_json.get("distinguishing_rule"),
        }

    elif content_type == "orbit_plot":
        structured = {
            "orbit_shape": caption_json.get("orbit_shape"),
            "fault_type": caption_json.get("fault_type"),
            "distinguishing_rule": caption_json.get("distinguishing_rule"),
        }

    elif content_type == "severity_chart":
        structured = {
            "standard": caption_json.get("standard"),
            "groups": caption_json.get("groups"),
            "zone_meanings": caption_json.get("zone_meanings"),
            "measurement_unit": caption_json.get("measurement_unit"),
            "frequency_range": caption_json.get("frequency_range"),
        }

    fault_type_for_check = structured.get("fault_type")
    generated_text = caption_json.get("description", "")

    review_caption = not _cross_validate_fault_type(fault_type_for_check, source_caption)
    review_orbit = (content_type == "orbit_plot" and
                    _orbit_consistency_flag(structured.get("orbit_shape"), fault_type_for_check))
    review_leak = (content_type == "spectrum" and
                _spectrum_diagram_leak_flag(generated_text))
    review_flag = review_caption or review_orbit or review_leak

    return VisualChunk(
        chunk_id=f"fig_p{page_number}_{prefix}_{local_index}",
        content_type=content_type,
        page_number=page_number,
        section_path=section_path,
        source_caption=source_caption,
        generated_text=caption_json.get("description", ""),
        structured_data=structured,
        confidence=caption_json.get("confidence", "unknown"),
        needs_review=review_flag,
    )


def run_ingestion_pipeline(pdf_path: str, output_path: str = "data/processed/visual_chunks.json"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY ortam değişkeni tanımlı değil")

    client = genai.Client(api_key=api_key)
    doc = fitz.open(pdf_path)

    print("PDF başlık hiyerarşisi çıkarılıyor...")
    headings = _extract_section_headings(doc)

    print("1) PDF'ten görseller çıkarılıyor...")
    raw_figures = extract_figure_images(doc=doc)
    print(f"   {len(raw_figures)} görsel bulundu")

    filtered_figures = [
        fig for fig in raw_figures
        if is_page_allowlisted(fig["page_number"], fig["local_index"])
    ]
    skipped_count = len(raw_figures) - len(filtered_figures)
    print(f"   {len(filtered_figures)} görsel allow-list'te, "
        f"{skipped_count} görsel LLM'e gönderilmeden atlandı")

    ingested_chunks: list[VisualChunk] = []
    failed_figures: list[dict] = []
    preflight_skipped: int = 0

    cached_page_num: int | None = None
    cached_page: fitz.Page | None = None

    for fig in filtered_figures:
        page_num = fig["page_number"]
        idx = fig["local_index"]
        mime = f"image/{fig['image_ext']}"
        xref = fig["xref"]

        if page_num != cached_page_num:
            cached_page = doc[page_num - 1]
            cached_page_num = page_num
        page = cached_page
        assert page is not None, f"cached_page beklenmedik biçimde None (sayfa {page_num})"

        rects = page.get_image_rects(xref)
        y_pos = rects[0].y0 if rects else 0.0
        current_section = _resolve_section_path(headings, page_num, y_pos)

        prompt, caption_text = classify_and_select_prompt(page)

        if _pre_flight_skip(caption_text, fig["image_bytes"]):
            preflight_skipped += 1
            print(f"   Sayfa {page_num}, görsel {idx} -> pre-flight skip (LLM çağrılmadı)")
            continue
        prompt_label, schema_cls = prompt_label_and_schema(prompt)

        print(f"2) Sayfa {page_num}, görsel {idx} [{prompt_label}] "
            f"caption'lanıyor...")

        try:
            caption_json = caption_figure_with_vision_llm(
                client, fig["image_bytes"], prompt=prompt, mime_type=mime,
                response_schema=schema_cls,
            )
        except Exception as e:
            print(f"   -> HATA, atlandı ve loglandı: {e}")
            failed_figures.append({
                "page": page_num, "index": idx,
                "prompt_label": prompt_label, "error": str(e),
            })
            continue

        if not is_worth_ingesting(caption_json):
            ct = caption_json.get("content_type", "?")
            print(f"   -> {ct}, atlandı")
            continue

        chunk = build_chunk(page_num, idx, caption_json,
                            source_caption=caption_text,
                            section_path=current_section,
                            prefix="vision")
        ingested_chunks.append(chunk)

        ct = chunk.content_type
        sd = chunk.structured_data or {}
        review_note = " [REVIEW GEREKLİ]" if chunk.needs_review else ""
        print(f"   -> content_type={ct}, confidence={chunk.confidence}{review_note}")
        if ct == "spectrum":
            if sd.get("domain") == "time":
                print(f"      fault_type={sd.get('fault_type')}, "
                    f"time_pattern={sd.get('time_domain_pattern')}")
            else:
                print(f"      fault_type={sd.get('fault_type')}, "
                    f"peaks={sd.get('peaks')}")
        elif ct == "bearing_stage":
            print(f"      stage={sd.get('stage')}, "
                f"hfd_trend={sd.get('hfd_amplitude_trend')}")
        elif ct == "orbit_plot":
            print(f"      orbit_shape={sd.get('orbit_shape')}, "
                f"fault_type={sd.get('fault_type')}")
        elif ct == "severity_chart":
            print(f"      standard={sd.get('standard')}, "
                f"groups={len(sd.get('groups') or [])}")

    doc.close()

    print(f"\n3) Visual chunklar {output_path} dosyasına kaydediliyor...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in ingested_chunks], f,
                ensure_ascii=False, indent=2)

    if failed_figures:
        fail_path = output_path.replace(".json", "_failed.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failed_figures, f, ensure_ascii=False, indent=2)
        print(f"   {len(failed_figures)} görsel başarısız oldu -> {fail_path}")

    review_needed = [c for c in ingested_chunks if c.needs_review]
    type_counts: dict[str, int] = {}
    for c in ingested_chunks:
        type_counts[c.content_type] = type_counts.get(c.content_type, 0) + 1

    print(f"\nBitti. {len(ingested_chunks)} chunk işlendi:")
    for ct, count in sorted(type_counts.items()):
        print(f"   {ct}: {count}")
    if preflight_skipped > 0:
        print(f"   Pre-flight skip (LLM çağrılmadı): {preflight_skipped}")
    print(f"   Review gereken chunk sayısı: {len(review_needed)}")
    if review_needed:
        for c in review_needed:
            print(f"     - {c.chunk_id} (fault_type uyuşmazlığı olabilir)")
    print(f"Çıktı: {output_path}")


if __name__ == "__main__":
    run_ingestion_pipeline(pdf_path="data/raw/VibrationAnalysisandDiagnosticGuide.pdf", output_path="data/processed/visual_chunks.json")