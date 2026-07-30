
"""
Text Parser Modülü
================================================================

Amaç: PDF sayfalarından, tablo ve görsel (figür) alanlarını
dışarıda bırakarak saf paragraf metnini çıkarmak; section_utils
ile hiyerarşik başlıklara bağlamak; ve RAG için parent/child
chunk çifti üretmek.

Pipeline akışı (bu modül adım 2'yi karşılar):

    PDF Sayfası
    │
    ├──> 1. Table & Vision Pipeline (önce çalışır — table_extraction.py,
    │       vision_pipeline.py)
    │       └── Çıktı: Table/Image chunk'lar + bbox listesi [x0,y0,x1,y1]
    │
    └──> 2. Text Parser (bu modül)
            ├── Bbox listesindeki alanları MASKELER (skip eder)
            ├── Kalan metin bloklarını section_utils ile BAŞLIKLARA bağlar
            ├── Bölüm bazlı PARENT chunk'ları üretir (~1200 token)
            └── Parent'ları bölerek CHILD chunk'ları üretir (~300 token)

Not — bbox kaynağı:
    table_extraction.TableChunk şu an bbox saklamıyor (sadece page_number).
    Bu yüzden build_masked_bbox_map() burada page.find_tables() ile aynı
    native tespiti (table_extraction.py ile birebir aynı ghost-table
    filtresini kullanarak) tekrar çalıştırıp bbox üretir; sonuçlar bu
    sayede tutarlı kalır. Görseller (raster image) get_text("blocks")
    çıktısında zaten ayrı bir block_type (=1) ile geldiğinden, metin
    blokları (block_type=0) alınırken otomatik olarak dışarıda kalır —
    bu yüzden figürler için ayrıca maskeleme gerekmez; yine de dışarıdan
    (vision_pipeline.extract_figure_images çıktısı) bbox verilirse bunlar
    da maskeye eklenir (ekstra güvenlik).

Gereksinimler:
    pip install pymupdf --break-system-packages
    (tiktoken opsiyonel: daha doğru token sayımı için)
"""

import re
from dataclasses import dataclass
from typing import Optional
import fitz
from ingestion.section_utils import (
    _extract_section_headings,
    _resolve_section_path,
)
from ingestion.table_extraction import _is_ghost_table
from config_loader import load_retcfg, load_prompts_cfg
from schemas.schemas import TextChunk

ret_cfg = load_retcfg()
prompts_cfg = load_prompts_cfg()

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except ImportError:
    def _count_tokens(text: str) -> int:
        # tiktoken yoksa kaba yaklaşım: ~4 karakter = 1 token
        return max(1, len(text) // 4)


_chunking_cfg = ret_cfg.parent_child_retrieval.get("chunking", {})

PARENT_TOKEN_TARGET = _chunking_cfg.get("parent_chunk_size", 1500)
CHILD_TOKEN_TARGET = _chunking_cfg.get("child_chunk_size", 200)
CHILD_TOKEN_OVERLAP = _chunking_cfg.get("child_chunk_overlap", 20)
MIN_GROUP_TOKENS = _chunking_cfg.get("min_group_tokens", 150)


def build_masked_bbox_map(pdf_path: str,
                        raw_figures: Optional[list[dict]] = None
                        ) -> dict[int, list[tuple]]:
    """
    Table & Vision pipeline'ının kapsadığı alanları sayfa bazlı bbox
    sözlüğüne derler: {page_number: [(x0,y0,x1,y1), ...]}

    Tablolar: page.find_tables() + table_extraction._is_ghost_table
    filtresi ile (ghost/false-positive tablolar maskeye dahil edilmez,
    aksi halde gerçek paragraf metni yanlışlıkla atılabilir).

    Görseller: raw_figures verilirse (vision_pipeline.extract_figure_images
    çıktısı, xref içerir) sayfadaki gömülü görsellerin gerçek dikdörtgenleri
    de eklenir. Bu genelde gerekli değildir çünkü get_text("blocks")
    zaten raster görselleri ayrı block_type ile döner; yine de örtüşen
    caption/metin bloklarını daha temiz ayırmak için opsiyonel bir
    güvenlik katmanı olarak sunulur.
    """
    doc = fitz.open(pdf_path)
    bbox_map: dict[int, list[tuple]] = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        rects: list[tuple] = []

        try:
            table_finder = page.find_tables()
            for t in table_finder.tables:
                data = t.extract()
                if data and not _is_ghost_table(data):
                    rects.append(tuple(t.bbox))
        except Exception:
            pass

        if rects:
            bbox_map[page_number] = rects

    if raw_figures:
        for fig in raw_figures:
            page_number = fig.get("page_number")
            xref = fig.get("xref")
            if page_number is None or xref is None:
                continue
            page = doc[page_number - 1]
            for r in page.get_image_rects(xref):
                bbox_map.setdefault(page_number, []).append(tuple(r))

    doc.close()
    return bbox_map


def _block_is_masked(block_rect: fitz.Rect, masked_rects: list[fitz.Rect],
                    overlap_threshold: float = 0.5) -> bool:
    """
    Bir metin bloğu, maskelenmiş alanla yeterince örtüşüyorsa (varsayılan
    >%50) atılır. Basit intersects() yerine örtüşme ORANI kullanılır ki
    tablonun hemen üstündeki/altındaki başlık/paragraf metni (kenardan
    hafifçe değen bloklar) yanlışlıkla silinmesin.
    """
    block_area = block_rect.get_area()
    if block_area <= 0:
        return False

    for rect in masked_rects:
        inter = block_rect & rect
        if inter.is_empty:
            continue
        if inter.get_area() / block_area >= overlap_threshold:
            return True
    return False



def extract_masked_text_blocks(pdf_path: str,
                                masked_bboxes_by_page: Optional[dict[int, list[tuple]]] = None
                                ) -> list[dict]:
    """
    PDF'teki tüm sayfalardan tablo/görsel alanlarını maskeleyerek saf
    metin bloklarını çıkarır ve her bloğu section_utils ile hiyerarşik
    başlığa (section_path) bağlar.

    Args:
        pdf_path: PDF dosya yolu
        masked_bboxes_by_page: {page_number: [(x0,y0,x1,y1), ...]}
            Table & Vision pipeline'ından (build_masked_bbox_map ile
            veya doğrudan pipeline çıktılarından) derlenen bbox listesi.
            None verilirse, bu fonksiyon build_masked_bbox_map'i kendi
            içinde çağırarak fallback üretir (bağımsız kullanım için).

    Returns:
        Sayfa + düşey konuma göre okuma sırasına sıralı blok listesi:
        [{"page": int, "y0": float, "text": str, "section_path": str}, ...]
    """
    if masked_bboxes_by_page is None:
        masked_bboxes_by_page = build_masked_bbox_map(pdf_path)

    doc = fitz.open(pdf_path)
    headings = _extract_section_headings(doc)

    blocks_out: list[dict] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        masked_rects = [
            fitz.Rect(b) for b in masked_bboxes_by_page.get(page_number, [])
        ]

        raw_blocks = page.get_text("blocks")
        raw_blocks = sorted(raw_blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

        for b in raw_blocks:
            block_type = b[6]
            if block_type != 0:  
                continue

            block_rect = fitz.Rect(b[:4])
            block_text = b[4].strip()
            if not block_text:
                continue

            if _block_is_masked(block_rect, masked_rects):
                continue

            block_text = re.sub(r"[ \t]+", " ", block_text)
            block_text = re.sub(r"\n{3,}", "\n\n", block_text)

            section_path = _resolve_section_path(headings, page_number, block_rect.y0)

            blocks_out.append({
                "page": page_number,
                "y0": block_rect.y0,
                "text": block_text,
                "section_path": section_path,
            })

    doc.close()
    return blocks_out


def _group_blocks_by_section(blocks: list[dict]) -> list[dict]:
    """
    Ardışık blokları aynı section_path'e göre gruplar (okuma sırası
    korunur). Bir grup = bir bölümün (veya bölüm parçasının) tüm metni.
    """
    groups: list[dict] = []
    current_section = None
    current_texts: list[str] = []
    current_pages: set[int] = set()

    def _flush():
        if current_texts:
            groups.append({
                "section_path": current_section,
                "text": "\n\n".join(current_texts),
                "pages": sorted(current_pages),
            })

    for blk in blocks:
        if blk["section_path"] != current_section:
            _flush()
            current_section = blk["section_path"]
            current_texts = []
            current_pages = set()
        current_texts.append(blk["text"])
        current_pages.add(blk["page"])

    _flush()
    return groups


def _merge_small_groups(groups: list[dict],
                        min_tokens: int = MIN_GROUP_TOKENS) -> list[dict]:
    """
    min_tokens'tan küçük section grupları (örn. tek cümlelik alt
    başlıklar) bir sonraki grupla birleştirilir; aksi halde anlamsız
    derecede küçük parent chunk'lar üretilir. İlk section_path korunur
    (parent, o bölümün başlangıcı sayılır).
    """
    if not groups:
        return groups

    merged: list[dict] = []
    buffer: Optional[dict] = None

    for g in groups:
        if buffer is None:
            buffer = dict(g)
            continue

        if _count_tokens(buffer["text"]) < min_tokens:
            buffer["text"] += "\n\n" + g["text"]
            buffer["pages"] = sorted(set(buffer["pages"]) | set(g["pages"]))
            continue

        merged.append(buffer)
        buffer = dict(g)

    if buffer is not None:
        merged.append(buffer)

    return merged

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_TOKEN_TARGET,
    chunk_overlap=0,
    length_function=_count_tokens,
    separators=["\n\n", "\n", ". ", " ", ""]
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_TOKEN_TARGET,
    chunk_overlap=CHILD_TOKEN_OVERLAP,
    length_function=_count_tokens,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def build_chunks(pdf_path: str,
                masked_bboxes_by_page: Optional[dict[int, list[tuple]]] = None,
                chunk_id_prefix: str = "text") -> list[TextChunk]:
    """
    PDF'ten parent/child metin chunk'larını üretir.

    Adımlar:
        1. extract_masked_text_blocks: tablo (ve varsa görsel) alanları
        maskelenmiş ham metin blokları + section_path
        2. _group_blocks_by_section: ardışık blokları bölüme göre grupla
        3. _merge_small_groups: çok küçük grupları komşusuyla birleştir
        4. Her grup için PARENT chunk'lar üret (~1200 token; grup daha
        büyükse birden çok parent'a bölünür)
        5. Her parent'ı CHILD chunk'lara böl (~300 token, 50 token overlap)

    Returns:
        TextChunk listesi (parent ve child seviyeleri karışık olarak;
        chunk_level alanıyla ayırt edilir, child'lar parent_id ile bağlanır)
    """
    blocks = extract_masked_text_blocks(pdf_path, masked_bboxes_by_page)
    groups = _merge_small_groups(_group_blocks_by_section(blocks))

    all_chunks: list[TextChunk] = []
    parent_counter = 0

    for group in groups:
        section_path = group["section_path"]
        pages = group["pages"]
        page_number = pages[0] if pages else 0
        page_range = pages if len(pages) > 1 else None

        parent_texts = parent_splitter.split_text(group["text"])

        for p_text in parent_texts:
            parent_counter += 1
            parent_id = f"{chunk_id_prefix}_parent_{parent_counter}"

            all_chunks.append(TextChunk(
                chunk_id=parent_id,
                chunk_level="parent",
                parent_id=None,
                page_number=page_number,
                page_range=page_range,
                section_path=section_path,
                text=p_text,
                token_count=_count_tokens(p_text),
            ))

            child_texts = child_splitter.split_text(p_text)

            for c_idx, c_text in enumerate(child_texts):
                all_chunks.append(TextChunk(
                    chunk_id=f"{parent_id}_child_{c_idx}",
                    chunk_level="child",
                    parent_id=parent_id,
                    page_number=page_number,
                    page_range=page_range,
                    section_path=section_path,
                    text=c_text,
                    token_count=_count_tokens(c_text),
                ))

    return all_chunks


def run_text_extraction(pdf_path: str,
                        output_path: str = "text_chunks.json",
                        masked_bboxes_by_page: Optional[dict[int, list[tuple]]] = None):
    """
    Text extraction pipeline'ını çalıştırır ve sonucu JSON'a yazar.

    masked_bboxes_by_page verilmezse, table_extraction.py'daki native
    tablo tespitiyle tutarlı bir maske otomatik üretilir (bkz.
    build_masked_bbox_map). Gerçek üretim akışında bu parametreye,
    table_extraction.extract_all_tables() çalıştıktan sonra elde
    edilen bbox listesini vermek tercih edilir ki tablo/vision
    pipeline ile text parser aynı maskeyi kullansın.
    """
    import json
    from dataclasses import asdict

    print("1) Tablo alanları tespit ediliyor (maske için)...")
    if masked_bboxes_by_page is None:
        masked_bboxes_by_page = build_masked_bbox_map(pdf_path)
    total_masked = sum(len(v) for v in masked_bboxes_by_page.values())
    print(f"   {total_masked} alan maskelendi "
        f"({len(masked_bboxes_by_page)} sayfada)")

    print("2) Metin çıkarılıyor, section'lara bağlanıyor, parent/child'a bölünüyor...")
    chunks = build_chunks(pdf_path, masked_bboxes_by_page)

    parents = [c for c in chunks if c.chunk_level == "parent"]
    children = [c for c in chunks if c.chunk_level == "child"]

    print(f"\n{'═' * 50}")
    print(f"  {len(parents)} parent chunk, {len(children)} child chunk üretildi")
    for p in parents[:5]:
        print(f"    {p.chunk_id} [{p.section_path}] ~{p.token_count} token")
    if len(parents) > 5:
        print(f"    ... (+{len(parents) - 5} tane daha)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print(f"\nKaydedildi -> {output_path}")
    return chunks


if __name__ == "__main__":
    run_text_extraction("data/raw/VibrationAnalysisandDiagnosticGuide.pdf")
