
import fitz
import re
from typing import Optional

_HEADING_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+')


def _extract_section_headings(doc: fitz.Document) -> list[dict]:
    """
    PDF'teki tüm section başlıklarını font analizi ile tespit eder.

    Bu PDF'te TOC (bookmark) tanımlı olmadığından, başlıklar şu kuralla
    tespit edilir:
        - Font boyutu >= 11.0pt
        - Bold flag aktif (flags & 16)
        - Numaralı pattern ile başlıyor: ``^\\d+(\\.\\d+)*\\s``

    Returns:
        Sıralı liste, her eleman:
        {
            "page": int,       # 1-indexed sayfa numarası
            "y": float,        # satırın y pozisyonu (origin)
            "level": int,      # hierarşi seviyesi (1=ana bölüm, 2=alt, 3=alt-alt)
            "number": str,     # bölüm numarası (ör. "4.11")
            "text": str,       # tam başlık metni
        }
    """
    headings: list[dict] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                full_text = "".join(span["text"] for span in spans).strip()
                if not full_text or len(full_text) < 3:
                    continue

                max_size = max(round(span["size"], 1) for span in spans)
                is_bold = any(span["flags"] & 16 for span in spans)

                if max_size < 11.0 or not is_bold:
                    continue

                match = _HEADING_PATTERN.match(full_text)
                if not match:
                    continue

                number = match.group(1)
                level = number.count(".") + 1

                headings.append({
                    "page": page_num,
                    "y": spans[0]["origin"][1],
                    "level": level,
                    "number": number,
                    "text": full_text.strip(),
                })

    return headings


def _resolve_section_path(
    headings: list[dict],
    page: int,
    y_pos: float = 0.0,
) -> str:
    """
    Verilen sayfa ve y pozisyonu için hierarşik section path hesaplar.

    Algoritma:
        1. Hedef pozisyondan önce gelen tüm başlıkları filtrele
        2. En son karşılaşılan başlığın numarasını al (ör. "6.2.2")
        3. Number prefix chain ile parent başlıkları bul:
        "6.2.2" → "6.2" → "6" → tam path
        4. "6. Advanced Techniques > 6.2 Dual... > 6.2.2 Relative..." döndür

    Args:
        headings: _extract_section_headings() çıktısı
        page: 1-indexed tablo sayfa numarası
        y_pos: tablonun üst kenarının y koordinatı (0 = sayfa üstü)

    Returns:
        Hierarşik section path string. Başlık bulunamazsa "unknown".
    """
    if not headings:
        return "unknown"

    preceding = [
        h for h in headings
        if h["page"] < page or (h["page"] == page and h["y"] < y_pos)
    ]

    if not preceding:
        return "unknown"

    last = preceding[-1]
    target_number = last["number"]

    parts = target_number.split(".")
    chain: list[str] = []

    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])
        # Bu prefix'e sahip en son başlığı bul
        matches = [h for h in preceding if h["number"] == prefix]
        if matches:
            chain.append(matches[-1]["text"])

    if not chain:
        return last["text"]

    return " > ".join(chain)