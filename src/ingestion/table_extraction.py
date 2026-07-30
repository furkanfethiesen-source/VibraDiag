
"""
Tablo Extraction Modülü
================================================================

PDF'teki native tablolarını PyMuPDF find_tables() ile çıkarır.
Güvenilmez veya tespit edilemeyen tablolar için Gemini vision LLM
fallback uygular.

Sorun tipleri ve çözümleri:
  - Ghost table (false positive)   → _is_ghost_table filtresi
  - Merged-cell / staircase        → Vision LLM fallback
  - Newline-in-cell               → _clean_cell temizliği
  - Borderless table (tespit yok) → Sayfa renderla → vision LLM
  - Cross-page split              → _merge_cross_page_tables
  - Severity chart (native text)  → vision_pipeline SEVERITY_CHART_PROMPT
  - Section path detection         → Font analizi ile bölüm başlığı tespiti

Gereksinimler:
    pip install pymupdf google-genai --break-system-packages

Ortam değişkeni (vision LLM fallback için, opsiyonel):
    export GEMINI_API_KEY="..."
"""

import os
import fitz
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional
from dotenv import load_dotenv
from ingestion.section_utils import (
    _extract_section_headings,
    _resolve_section_path
)
from ingestion.vision_pipeline import (
    caption_figure_with_vision_llm,
    SEVERITY_CHART_PROMPT,
    build_chunk,
)
from config_loader import load_appcfg, load_prompts_cfg
from schemas.schemas import TableChunk

load_dotenv()

app_cfg = load_appcfg()
prompts_cfg = load_prompts_cfg()

VISION_TABLE_PROMPT = prompts_cfg.parsing_prompts.get("vision_table_prompt", "")

def _clean_cell(value) -> str:
    """Hücre değerini temizler: None→'', \\n→' ', strip."""
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _is_ghost_table(data: list[list]) -> bool:
    """
    False positive tablo tespiti (figür annotation, formül blokları vb.).
    Kriterler:
    - Toplam hücre < 4
    - Dolu hücre sayısı < 3
    - Dolu hücre oranı < %35
    - Veri satırı (header hariç) < 1
    """
    if not data:
        return True

    total_cells = sum(len(row) for row in data)
    if total_cells < 4:
        return True

    filled = sum(
        1 for row in data for c in row
        if c is not None and str(c).strip()
    )

    if filled < 3:
        return True

    if total_cells > 0 and filled / total_cells < 0.35:
        return True

    if len(data) < 2:
        return True

    return False


def _none_ratio(rows: list[list]) -> float:
    """Satırlardaki None/boş hücre oranını hesaplar."""
    total = sum(len(row) for row in rows)
    if total == 0:
        return 1.0
    none_count = sum(
        1 for row in rows for c in row
        if c is None or (isinstance(c, str) and not c.strip())
    )
    return none_count / total


def _has_severity_keywords(title: str, data: list[list]) -> bool:
    """Tablonun severity chart olup olmadığını keyword ile tespit eder."""
    keywords = {"severity", "iso 10816", "iso10816", "vdi 2056",
                "vdi2056", "ird mechanalysis"}
    text = title.lower()
    for row in data[:3]:
        for cell in row:
            if cell:
                text += " " + str(cell).lower()
    return any(kw in text for kw in keywords)


def _guess_table_title(page: fitz.Page, bbox: tuple) -> str:
    """
    Tablonun üstündeki veya altındaki 'Table X.Y ...' başlığını bulur.
    Bulunamazsa en yakın metni döner.
    """
    above = fitz.Rect(
        bbox[0], max(0, bbox[1] - 60), bbox[2], bbox[1]
    )
    above_text = page.get_textbox(above).strip()

    match = re.search(r'Table\s+\d+\.\d+[^\n]*', above_text)
    if match:
        return match.group(0).strip()

    below = fitz.Rect(
        bbox[0], bbox[3],
        bbox[2], min(page.rect.height, bbox[3] + 80)
    )
    below_text = page.get_textbox(below).strip()

    match = re.search(r'(?:Table|Figure)\s+\d+\.\d+[^\n]*', below_text)
    if match:
        return match.group(0).strip()

    if above_text:
        return above_text[:100]
    if below_text:
        return below_text[:100]
    return "(başlık bulunamadı)"


_HEADING_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+')

def _to_markdown(headers: list, rows: list) -> str:
    """
    Embedding modeline ve agent context'ine verilecek markdown tablo.
    Hücre içi \\n karakterleri temizlenir.
    """
    clean_headers = [_clean_cell(h) for h in headers]
    header_line = "| " + " | ".join(clean_headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in clean_headers) + " |"
    row_lines = [
        "| " + " | ".join(_clean_cell(c) for c in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)



def is_native_table_extraction_reliable(chunk: TableChunk) -> bool:
    """
    find_tables() çıktısının güvenilirliğini çok boyutlu kontrol eder:
    1. Sütun sayısı tutarlılığı
    2. None/boş hücre yoğunluğu (veri satırlarında >%50 → unreliable)
    3. Minimum dolu hücre sayısı
    4. Header kalitesi (>%50 None/boş → unreliable)
    """
    if not chunk.headers or not chunk.rows:
        return False

    expected_cols = len(chunk.headers)

    if not all(len(row) == expected_cols for row in chunk.rows):
        return False

    if _none_ratio(chunk.rows) > 0.50:
        return False

    filled = sum(
        1 for row in chunk.rows for c in row
        if c is not None and str(c).strip()
    )
    if filled < 4:
        return False

    header_filled = sum(
        1 for h in chunk.headers
        if h is not None and str(h).strip()
    )
    if header_filled < len(chunk.headers) * 0.5:
        return False

    return True



def _find_pages_with_missing_tables(doc: fitz.Document,
                                    found_table_pages: set[int]) -> list[int]:
    """
    Sayfada 'Table X.Y' caption'ı var ama find_tables() tablo bulamadı
    → borderless table adayı.
    """
    missing = []
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num in found_table_pages:
            continue
        text = doc[page_idx].get_text()
        if re.search(r'Table\s+\d+\.\d+\s+[A-Z]', text):
            missing.append(page_num)
    return missing



def _merge_cross_page_tables(chunks: list[TableChunk]) -> list[TableChunk]:
    """
    Ardışık sayfalardaki aynı sütun yapısına sahip tabloları birleştirir.

    Merge koşulları:
    1. Ardışık sayfalar (page N → page N+1)
    2. Aynı sütun sayısı
    3. Sonraki tablonun kendi 'Table X.Y' başlığı yok (yani devam tablosu)

    Not: Şu an sadece 2-sayfa merge destekliyor. 3+ sayfa zincirleri
    için iteratif çağrı gerekir.
    """
    if len(chunks) < 2:
        return chunks

    merged = []
    skip_indices: set[int] = set()

    for i, chunk in enumerate(chunks):
        if i in skip_indices:
            continue

        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]

            if (next_chunk.page_number == chunk.page_number + 1
                    and chunk.headers and next_chunk.headers
                    and len(chunk.headers) == len(next_chunk.headers)):

                next_has_own_title = bool(
                    re.search(r'Table\s+\d+\.\d+', next_chunk.table_title)
                )

                if not next_has_own_title:
                    if chunk.headers == next_chunk.headers:
                        merged_rows = (chunk.rows or []) + (next_chunk.rows or [])
                    else:
                        merged_rows = (
                            (chunk.rows or [])
                            + [next_chunk.headers]
                            + (next_chunk.rows or [])
                        )

                    chunk.rows = merged_rows
                    chunk.markdown_repr = _to_markdown(chunk.headers, merged_rows)
                    chunk.chunk_id = f"{chunk.chunk_id}+p{next_chunk.page_number}"
                    skip_indices.add(i + 1)

        merged.append(chunk)

    return merged






def _fallback_vision_table(client, page: fitz.Page,
                           page_number: int,
                           section_path: str = "unknown") -> Optional[TableChunk]:
    """
    Sayfayı renderleyip vision LLM'e gönderir. JSON yanıttan TableChunk
    oluşturur. client None ise atlar.
    """
    if client is None:
        return None

    pix = page.get_pixmap(dpi=200)
    image_bytes = pix.tobytes("png")

    print(f"   Vision LLM fallback (tablo): sayfa {page_number}...")

    try:
        caption_json = caption_figure_with_vision_llm(
            client, image_bytes,
            prompt=VISION_TABLE_PROMPT,
            mime_type="image/png",
        )
    except Exception as e:
        print(f"   Vision LLM fallback başarısız: {e}")
        return None

    headers = caption_json.get("headers", [])
    rows = caption_json.get("rows", [])
    title = caption_json.get("table_title", "")

    if not headers or not rows:
        return None

    return TableChunk(
        chunk_id=f"table_p{page_number}_vision",
        page_number=page_number,
        section_path=section_path,
        table_title=title,
        headers=headers,
        rows=rows,
        markdown_repr=_to_markdown(headers, rows),
    )


def _fallback_severity_chart(client, page: fitz.Page,
                             page_number: int,
                             section_path: str = "unknown",
                             prefix: str = "table") -> Optional[dict]:
    """
    Severity chart'ı vision LLM ile çıkarır. vision_pipeline'daki
    SEVERITY_CHART_PROMPT ve build_chunk kullanılır.
    Sonucu dict olarak döner (VisualChunk.asdict formatında).
    """
    if client is None:
        return None


    pix = page.get_pixmap(dpi=200)
    image_bytes = pix.tobytes("png")

    print(f"   Severity chart fallback: sayfa {page_number}...")

    try:
        caption_json = caption_figure_with_vision_llm(
            client, image_bytes,
            prompt=SEVERITY_CHART_PROMPT,
            mime_type="image/png",
        )
    except Exception as e:
        print(f"   Severity chart fallback başarısız: {e}")
        return None

    caption = _guess_table_title(
        page, (0, 0, page.rect.width, page.rect.height)
    )

    chunk = build_chunk(
        page_number, 0, caption_json,
        source_caption=caption,
        section_path=section_path,
    )
    return asdict(chunk)



def extract_all_tables(pdf_path: str,
                       client=None) -> tuple[list[TableChunk], list[dict]]:
    """
    PDF'teki tüm tabloları çıkarır:
    1. Section başlıklarını font analizi ile tespit et
    2. find_tables() ile native tablo çıkarma + ghost filtresi
    3. Güvenilirlik kontrolü → unreliable ise vision LLM fallback
    4. Borderless table tespiti → vision LLM fallback
    5. Cross-page merge
    6. Her chunk'a hierarşik section_path ata

    Args:
        pdf_path: PDF dosya yolu
        client: google.genai.Client (opsiyonel, vision fallback için)

    Returns:
        (table_chunks, visual_chunks)
        - table_chunks: TableChunk listesi
        - visual_chunks: dict listesi (severity chart gibi VisualChunk'lar)
    """
    doc = fitz.open(pdf_path)
    table_chunks: list[TableChunk] = []
    visual_chunks: list[dict] = []
    found_table_pages: set[int] = set()

    headings = _extract_section_headings(doc)
    print(f"   {len(headings)} section başlığı tespit edildi (font analizi)")

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        table_finder = page.find_tables()

        for t_idx, table in enumerate(table_finder.tables):
            data = table.extract()
            if not data:
                continue

            if _is_ghost_table(data):
                continue

            found_table_pages.add(page_number)

            headers = data[0]
            rows = data[1:]
            title = _guess_table_title(page, table.bbox)

            table_y = table.bbox[1]
            section_path = _resolve_section_path(headings, page_number, table_y)

            chunk = TableChunk(
                chunk_id=f"table_p{page_number}_{t_idx}",
                page_number=page_number,
                section_path=section_path,
                table_title=title,
                headers=headers,
                rows=rows,
                markdown_repr=_to_markdown(headers, rows),
            )

            if is_native_table_extraction_reliable(chunk):
                table_chunks.append(chunk)
            else:
                if _has_severity_keywords(title, data):
                    vc = _fallback_severity_chart(
                        client, page, page_number,
                        section_path=section_path,
                    )
                    if vc:
                        visual_chunks.append(vc)
                        continue

                fallback = _fallback_vision_table(
                    client, page, page_number,
                    section_path=section_path,
                )
                if fallback:
                    table_chunks.append(fallback)
                else:
                    print(f"   Fallback yok, native kullanılıyor: sayfa {page_number}")
                    table_chunks.append(chunk)

    missing_pages = _find_pages_with_missing_tables(doc, found_table_pages)
    for page_num in missing_pages:
        page = doc[page_num - 1]
        print(f"   Borderless table adayı: sayfa {page_num}")
        section_path = _resolve_section_path(headings, page_num, 0.0)
        fallback = _fallback_vision_table(
            client, page, page_num,
            section_path=section_path,
        )
        if fallback:
            table_chunks.append(fallback)

    doc.close()

    table_chunks.sort(key=lambda c: (c.page_number, c.chunk_id))
    table_chunks = _merge_cross_page_tables(table_chunks)

    return table_chunks, visual_chunks



def chunk_to_embedding_text(chunk: TableChunk) -> str:
    """
    Tabloyu embedding'e verilecek metne çevirir. Başlığı ve markdown
    gösterimini birlikte veriyoruz ki hem semantik arama hem de
    agent'ın context'e aldığında satır/sütun ilişkisini görmesi sağlansın.
    """
    return f"{chunk.table_title}\n\n{chunk.markdown_repr}"



def run_table_extraction(pdf_path: str,
                         output_path: str = "table_chunks.json"):
    """
    Tablo extraction pipeline'ını çalıştırır.
    GEMINI_API_KEY varsa vision LLM fallback aktif olur.
    """
    client = None
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("Vision LLM fallback aktif (GEMINI_API_KEY bulundu)")
    else:
        print("UYARI: GEMINI_API_KEY yok — vision LLM fallback devre dışı")

    print("\nNative tablolar çıkarılıyor...")
    table_chunks, visual_chunks = extract_all_tables(pdf_path, client=client)

    print(f"\n{'═' * 50}")
    print(f"  {len(table_chunks)} tablo çıkarıldı:")
    for c in table_chunks:
        title = c.table_title[:55] if c.table_title else "(başlıksız)"
        rows = len(c.rows or [])
        cols = len(c.headers or [])
        print(f"    {c.chunk_id}: \"{title}\" ({rows}×{cols})")

    if visual_chunks:
        print(f"\n  {len(visual_chunks)} severity chart (vision pipeline):")
        for vc in visual_chunks:
            ct = vc.get("content_type", "?")
            sd = vc.get("structured_data") or {}
            print(f"    sayfa {vc.get('page_number')}: "
                  f"{ct}, standard={sd.get('standard', '?')}")

    all_output = [asdict(c) for c in table_chunks]
    if visual_chunks:
        all_output.extend(visual_chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)

    print(f"\nKaydedildi -> {output_path}")
    return table_chunks, visual_chunks


if __name__ == "__main__":
    run_table_extraction("data/raw/VibrationAnalysisandDiagnosticGuide.pdf")