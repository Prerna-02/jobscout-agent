"""Turn PDF/DOCX bytes into plain text, dropping text a human reader can't see.

Hidden text (white, microscopic, or Word's "hidden" formatting) is a known way to
smuggle keywords or instructions into a CV. It never reaches the LLM.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any, NamedTuple

import pdfplumber
from docx import Document
from docx.shared import RGBColor
from docx.table import Table
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from ._types import CVLimits, CVRejected, SourceFormat

_MIN_VISIBLE_FONT_PT = 2.0
_WHITE = RGBColor(0xFF, 0xFF, 0xFF)


class RawText(NamedTuple):
    text: str
    page_count: int | None
    dropped_hidden_chars: int


def detect_format(data: bytes) -> SourceFormat | None:
    """Identify the file by its content, not its extension."""
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            return None
    return None


def extract(data: bytes, fmt: SourceFormat, limits: CVLimits) -> RawText:
    if fmt == "pdf":
        return _extract_pdf(data, limits)
    return _extract_docx(data)


# --- PDF ---------------------------------------------------------------------


def _extract_pdf(data: bytes, limits: CVLimits) -> RawText:
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:  # pdfminer raises many unrelated types for bad files
        raise CVRejected(
            "unreadable", "This PDF could not be opened. It may be damaged or password-protected."
        ) from exc

    with pdf:
        page_count = len(pdf.pages)
        if page_count > limits.max_pages:
            raise CVRejected(
                "too_many_pages",
                f"The CV has {page_count} pages. The limit is {limits.max_pages}.",
            )
        pages: list[str] = []
        dropped = 0
        try:
            for page in pdf.pages:
                hidden = sum(1 for char in page.chars if _is_hidden_pdf_char(char))
                dropped += hidden
                visible = page.filter(_keep_pdf_object) if hidden else page
                pages.append(visible.extract_text() or "")
        except Exception as exc:
            raise CVRejected("unreadable", "Text could not be read from this PDF.") from exc

    return RawText("\n\n".join(pages), page_count, dropped)


def _keep_pdf_object(obj: dict[str, Any]) -> bool:
    return not (obj.get("object_type") == "char" and _is_hidden_pdf_char(obj))


def _is_hidden_pdf_char(char: dict[str, Any]) -> bool:
    size = char.get("size")
    if isinstance(size, (int, float)) and size < _MIN_VISIBLE_FONT_PT:
        return True
    return _is_white(char.get("non_stroking_color"))


def _is_white(color: Any) -> bool:
    """True for white in grayscale, RGB or CMYK, the colour spaces PDFs use for text."""
    if isinstance(color, (int, float)):
        components: tuple[float, ...] = (float(color),)
    elif (
        isinstance(color, (tuple, list))
        and color
        and all(isinstance(v, (int, float)) for v in color)
    ):
        components = tuple(float(v) for v in color)
    else:
        return False

    if len(components) == 1:
        return components[0] >= 0.95
    if len(components) == 3:
        return min(components) >= 0.95
    if len(components) == 4:
        return max(components) <= 0.05
    return False


# --- DOCX --------------------------------------------------------------------


def _extract_docx(data: bytes) -> RawText:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise CVRejected("unreadable", "This Word file could not be opened.") from exc

    lines: list[str] = []
    dropped = 0
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            text, hidden = _paragraph_text(block)
            dropped += hidden
            lines.append(text)
        elif isinstance(block, Table):
            for row_text, hidden in _table_rows(block):
                dropped += hidden
                lines.append(row_text)

    return RawText("\n".join(lines), None, dropped)


def _table_rows(table: Table) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for row in table.rows:
        cells: list[str] = []
        hidden_total = 0
        for cell in row.cells:
            parts = []
            for paragraph in cell.paragraphs:
                text, hidden = _paragraph_text(paragraph)
                hidden_total += hidden
                parts.append(text)
            cell_text = " ".join(p for p in parts if p).strip()
            # Merged cells repeat the same text once per grid column.
            if cell_text and (not cells or cells[-1] != cell_text):
                cells.append(cell_text)
        rows.append((" | ".join(cells), hidden_total))
    return rows


def _paragraph_text(paragraph: Paragraph) -> tuple[str, int]:
    visible: list[str] = []
    hidden = 0
    for item in paragraph.iter_inner_content():
        runs = item.runs if isinstance(item, Hyperlink) else [item]
        for run in runs:
            if _is_hidden_run(run):
                hidden += len(run.text)
            else:
                visible.append(run.text)
    return "".join(visible), hidden


def _is_hidden_run(run: Run) -> bool:
    font = run.font
    if font.hidden:
        return True
    if font.size is not None and font.size.pt < _MIN_VISIBLE_FONT_PT:
        return True
    try:
        return font.color is not None and font.color.rgb == _WHITE
    except (AttributeError, ValueError):
        return False
