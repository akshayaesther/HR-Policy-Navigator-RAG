"""Document loader and chunker for enterprise HR policy documents.

Implements PyMuPDF block-based extraction, two-column layout resolution,
table extraction via page.find_tables(), fallback OCR for scanned pages
via rapidocr-onnxruntime, header/footer cleaning, section-aware chunking,
and ingestion reporting.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import re
import uuid
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR

from backend.config import get_settings
from backend.models.document import DocumentChunk, IngestionReport, ExtractionMethod
from backend.ingestion.cleaner import (
    fix_hyphenated_line_breaks,
    normalize_whitespace,
    detect_and_remove_headers_footers,
    compute_quality_score,
)

SECTION_HEADING_REGEX = re.compile(
    r"^(?:Section\s+\d+[:\.]?|SECTION\s+\d+[:\.]?|\d+\.\s+[A-Z\s]{3,}|#{1,4}\s+|[A-Z\s]{5,}:)",
    re.MULTILINE,
)


class DocumentLoader:
    """Robust document loader and chunker for PDF and Markdown files."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._ocr: Optional[RapidOCR] = None

    @property
    def ocr(self) -> RapidOCR:
        """Lazy initialization of RapidOCR instance."""
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def load_and_chunk(self, file_path: Path) -> Tuple[List[DocumentChunk], IngestionReport]:
        """Load document from path, clean text, extract tables/OCR, and generate chunks."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in [".pdf", ".md"]:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Supported formats: .pdf, .md"
            )

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        else:
            return self._load_markdown(file_path)

    def _load_pdf(self, file_path: Path) -> Tuple[List[DocumentChunk], IngestionReport]:
        """Extract content from PDF with two-column sorting, tables, OCR, and cleaning."""
        # Handle zero-byte files gracefully
        if file_path.stat().st_size == 0:
            report = IngestionReport(
                source=file_path.name,
                total_pages=0,
                total_chunks=0,
                text_chunks=0,
                table_chunks=0,
                ocr_pages=0,
                average_quality=0.0,
                methods_breakdown={},
            )
            return [], report

        doc = fitz.open(file_path)
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            report = IngestionReport(
                source=file_path.name,
                total_pages=0,
                total_chunks=0,
                text_chunks=0,
                table_chunks=0,
                ocr_pages=0,
                average_quality=0.0,
                methods_breakdown={},
            )
            return [], report

        raw_page_texts: List[str] = []
        page_methods: List[ExtractionMethod] = []
        table_chunks: List[DocumentChunk] = []
        table_count = 0
        ocr_page_count = 0

        # Step 1: Per-page extraction (Tables, Two-Column blocks, or OCR fallback)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1a. Extract tables
            table_bboxes = []
            try:
                table_finder = page.find_tables()
                if table_finder and table_finder.tables:
                    for tbl in table_finder.tables:
                        table_bboxes.append(tbl.bbox)
                        table_rows = tbl.extract()
                        if table_rows:
                            formatted_table = self._format_table_as_markdown(table_rows)
                            if formatted_table.strip():
                                table_count += 1
                                t_chunk = DocumentChunk(
                                    id=f"{file_path.name}_p{page_num}_tbl{table_count}",
                                    content=formatted_table,
                                    source=file_path.name,
                                    page=page_num,
                                    category="hr_policy",
                                    section=self._detect_section_heading(formatted_table) or "Table Data",
                                    chunk_type="table",
                                    extraction_method="table",
                                    quality=1.0,
                                )
                                table_chunks.append(t_chunk)
            except Exception:
                # If find_tables encounters an unsupported layout, fall back to block text
                pass

            # 1b. Block-based text extraction with two-column sorting
            blocks = page.get_text("blocks")
            # Filter text blocks (block_type == 0) not inside extracted tables
            text_blocks = []
            for b in blocks:
                if b[6] != 0:
                    continue
                # Exclude if block is inside any extracted table
                if any(self._is_inside_bbox(b[:4], tbox) for tbox in table_bboxes):
                    continue
                text_blocks.append(b)

            # Sort blocks to handle multi-column reading order
            sorted_blocks = self._sort_blocks_multicol(text_blocks, page.rect.width)
            page_text = "\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())

            # 1c. Detect scanned page (< 50 printable characters)
            clean_check = page_text.strip()
            if len(clean_check) < 50:
                # Scanned page: fallback to RapidOCR
                ocr_text = self._run_ocr_on_page(page)
                if ocr_text.strip():
                    page_text = ocr_text
                    page_methods.append("ocr")
                    ocr_page_count += 1
                else:
                    page_methods.append("text")
            else:
                page_methods.append("text")

            raw_page_texts.append(page_text)

        doc.close()

        # Step 2: Global cleaning across pages (header/footer/page numbers)
        cleaned_page_texts = detect_and_remove_headers_footers(raw_page_texts)

        # Step 3: De-hyphenate and normalize whitespace
        processed_page_texts: List[str] = []
        page_qualities: List[float] = []

        for p_idx, p_text in enumerate(cleaned_page_texts):
            dehyphenated = fix_hyphenated_line_breaks(p_text)
            normalized = normalize_whitespace(dehyphenated)
            processed_page_texts.append(normalized)
            quality = compute_quality_score(normalized, extraction_method=page_methods[p_idx])
            page_qualities.append(quality)

        # Step 4: Chunking with section-heading awareness
        text_chunks: List[DocumentChunk] = []
        chunk_idx = 1
        current_section = "General"

        for p_idx, text in enumerate(processed_page_texts):
            page_num = p_idx + 1
            method = page_methods[p_idx]
            quality = page_qualities[p_idx]

            if not text.strip():
                continue

            page_chunks, current_section = self._chunk_page_text(
                text=text,
                source=file_path.name,
                page_num=page_num,
                method=method,
                quality=quality,
                current_section=current_section,
                start_chunk_idx=chunk_idx,
            )
            text_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        all_chunks = text_chunks + table_chunks
        # Sort chunks by page then id
        all_chunks.sort(key=lambda c: (c.page, c.chunk_type != "table", c.id))

        # Generate report
        methods_breakdown: Dict[str, int] = {}
        for m in page_methods:
            methods_breakdown[m] = methods_breakdown.get(m, 0) + 1
        if table_count > 0:
            methods_breakdown["table"] = table_count

        avg_quality = (
            round(sum(page_qualities) / len(page_qualities), 2)
            if page_qualities
            else 0.0
        )

        report = IngestionReport(
            source=file_path.name,
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            text_chunks=len(text_chunks),
            table_chunks=len(table_chunks),
            ocr_pages=ocr_page_count,
            average_quality=avg_quality,
            methods_breakdown=methods_breakdown,
        )

        return all_chunks, report

    def _load_markdown(self, file_path: Path) -> Tuple[List[DocumentChunk], IngestionReport]:
        """Load and chunk Markdown file."""
        content = file_path.read_text(encoding="utf-8")
        normalized = normalize_whitespace(fix_hyphenated_line_breaks(content))
        quality = compute_quality_score(normalized, extraction_method="text")

        chunks, _ = self._chunk_page_text(
            text=normalized,
            source=file_path.name,
            page_num=1,
            method="text",
            quality=quality,
            current_section="General",
            start_chunk_idx=1,
        )

        report = IngestionReport(
            source=file_path.name,
            total_pages=1,
            total_chunks=len(chunks),
            text_chunks=len(chunks),
            table_chunks=0,
            ocr_pages=0,
            average_quality=quality,
            methods_breakdown={"text": 1},
        )
        return chunks, report

    def _run_ocr_on_page(self, page: fitz.Page) -> str:
        """Render page as image and run RapidOCR."""
        try:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            result, _ = self.ocr(img_bytes)
            if not result:
                return ""
            lines = [line_info[1] for line_info in result if line_info and len(line_info) > 1]
            return "\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    def _is_inside_bbox(block_bbox: Tuple[float, float, float, float], table_bbox: Tuple[float, float, float, float]) -> bool:
        """Check if block is enclosed within or overlaps heavily with table bbox."""
        bx0, by0, bx1, by1 = block_bbox
        tx0, ty0, tx1, ty1 = table_bbox
        # Slight margin of error for table bounding boxes
        return (bx0 >= tx0 - 5) and (by0 >= ty0 - 5) and (bx1 <= tx1 + 5) and (by1 <= ty1 + 5)

    @staticmethod
    def _sort_blocks_multicol(blocks: List[Any], page_width: float) -> List[Any]:
        """Sort blocks into multi-column reading order (left column then right column)."""
        if not blocks:
            return []

        mid_x = page_width / 2.0
        # Check if page has blocks in both columns
        has_left = any(b[2] <= mid_x + 30 for b in blocks)
        has_right = any(b[0] >= mid_x - 30 for b in blocks)

        if not (has_left and has_right):
            # Single-column: sort vertically by y0
            return sorted(blocks, key=lambda b: b[1])

        # Multi-column sort
        def _col_sort_key(b: Any) -> Tuple[int, float, float]:
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            # Spans full width across columns
            if x0 < mid_x - 40 and x1 > mid_x + 40:
                # Top full-width banner
                if y0 < 120:
                    return (0, 0.0, y0)
                # Bottom full-width footer
                return (2, 0.0, y0)
            # Left column
            if x1 <= mid_x + 30:
                return (1, 0, y0)
            # Right column
            return (1, 1, y0)

        return sorted(blocks, key=_col_sort_key)

    @staticmethod
    def _format_table_as_markdown(rows: List[List[Optional[str]]]) -> str:
        """Convert 2D table array into readable Markdown table."""
        if not rows:
            return ""

        clean_rows = []
        for row in rows:
            clean_rows.append([str(cell or "").replace("\n", " ").strip() for cell in row])

        if not clean_rows:
            return ""

        headers = clean_rows[0]
        md_lines = ["| " + " | ".join(headers) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in clean_rows[1:]:
            # Pad row if incomplete
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            md_lines.append("| " + " | ".join(row[: len(headers)]) + " |")

        return "\n".join(md_lines)

    @staticmethod
    def _detect_section_heading(text: str) -> Optional[str]:
        """Detect and return first section heading found in text."""
        for line in text.split("\n"):
            line_str = line.strip()
            if SECTION_HEADING_REGEX.search(line_str):
                return line_str
        return None

    def _chunk_page_text(
        self,
        text: str,
        source: str,
        page_num: int,
        method: ExtractionMethod,
        quality: float,
        current_section: str,
        start_chunk_idx: int,
    ) -> Tuple[List[DocumentChunk], str]:
        """Chunk a single page of text with overlap and section heading propagation."""
        chunks: List[DocumentChunk] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_buffer = ""
        chunk_counter = start_chunk_idx

        for para in paragraphs:
            # Check if paragraph contains or begins with a section heading
            heading = self._detect_section_heading(para)
            if heading and current_buffer.strip():
                # Flush previous section content before starting new section
                chunk = DocumentChunk(
                    id=f"{source}_p{page_num}_c{chunk_counter}",
                    content=current_buffer.strip(),
                    source=source,
                    page=page_num,
                    category="hr_policy",
                    section=current_section,
                    chunk_type="text",
                    extraction_method=method,
                    quality=quality,
                )
                chunks.append(chunk)
                chunk_counter += 1
                current_buffer = ""
                current_section = heading
            elif heading:
                current_section = heading

            # If adding paragraph exceeds chunk_size, flush buffer
            if len(current_buffer) + len(para) + 2 > self.chunk_size and current_buffer:
                chunk = DocumentChunk(
                    id=f"{source}_p{page_num}_c{chunk_counter}",
                    content=current_buffer.strip(),
                    source=source,
                    page=page_num,
                    category="hr_policy",
                    section=current_section,
                    chunk_type="text",
                    extraction_method=method,
                    quality=quality,
                )
                chunks.append(chunk)
                chunk_counter += 1

                # Retain overlap from end of buffer
                overlap_text = current_buffer[-self.chunk_overlap :] if len(current_buffer) > self.chunk_overlap else ""
                current_buffer = (overlap_text + " " + para).strip()
            else:
                current_buffer = f"{current_buffer}\n\n{para}".strip() if current_buffer else para

        # Flush remaining buffer
        if current_buffer.strip():
            chunk = DocumentChunk(
                id=f"{source}_p{page_num}_c{chunk_counter}",
                content=current_buffer.strip(),
                source=source,
                page=page_num,
                category="hr_policy",
                section=current_section,
                chunk_type="text",
                extraction_method=method,
                quality=quality,
            )
            chunks.append(chunk)

        return chunks, current_section


def print_ingestion_report(report: IngestionReport) -> None:
    """Print structured ingestion summary per file."""
    methods_str = ", ".join(f"{k}: {v}" for k, v in report.methods_breakdown.items())
    print("\n" + "=" * 60)
    print(f" INGESTION REPORT: {report.source}")
    print("=" * 60)
    print(f"  Total Pages:        {report.total_pages}")
    print(f"  Total Chunks:       {report.total_chunks}")
    print(f"  Text Chunks:        {report.text_chunks}")
    print(f"  Table Chunks:       {report.table_chunks}")
    print(f"  OCR Pages:          {report.ocr_pages}")
    print(f"  Average Quality:    {report.average_quality:.2f}")
    print(f"  Extraction Methods: {methods_str}")
    print("=" * 60)
