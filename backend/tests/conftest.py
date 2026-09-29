from __future__ import annotations

import io
from collections.abc import Iterable

import pytest
from docx import Document
from docx.shared import RGBColor
from fpdf import FPDF

from jobscout import observability
from jobscout.settings import get_settings

# A synthetic CV. ASCII only, because fpdf2's built-in fonts are Latin-1.
SAMPLE_CV = [
    "Jane Doe",
    "Data Analyst | jane.doe@example.com | +91 98765 43210 | linkedin.com/in/janedoe",
    "Bengaluru, India",
    "Date of Birth: 12/03/1998 | Gender: Female",
    "",
    "SUMMARY",
    "Data analyst with 4 years of experience building dashboards and SQL pipelines "
    "for retail clients.",
    "",
    "SKILLS",
    "SQL, Python, PowerBI, Excel, Node.js, pandas, statistics, C++",
    "",
    "EXPERIENCE",
    "Data Analyst, Acme Retail - Jan 2022 - Present",
    "Built Power BI dashboards used by 40 store managers. Cleaned 10,000 daily orders with SQL.",
    "Junior Analyst, Beta Corp - Jun 2020 - Dec 2021",
    "Automated weekly Excel reports with Python and presented results to leadership.",
    "",
    "EDUCATION",
    "B.Sc. Statistics, Example University, 2016 - 2019",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="run tests that call real APIs")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live and real API keys")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_settings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Unit tests never read real keys from .env or send traces."""
    if "live" not in request.keywords:
        for var in (
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.setenv(var, "")
    get_settings.cache_clear()
    observability._client.cache_clear()
    yield
    get_settings.cache_clear()
    observability._client.cache_clear()


def make_docx(
    lines: Iterable[str], *, hidden: str | None = None, white: str | None = None
) -> bytes:
    document = Document()
    for line in lines:
        document.add_paragraph(line)
    if hidden or white:
        paragraph = document.add_paragraph("References available on request.")
        if hidden:
            paragraph.add_run(" " + hidden).font.hidden = True
        if white:
            paragraph.add_run(" " + white).font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_pdf(lines: Iterable[str], *, white_lines: Iterable[str] = (), pages: int = 1) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=10)
    for page in range(pages):
        pdf.add_page()
        if page == 0:
            for line in lines:
                pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(255, 255, 255)
            for line in white_lines:
                pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.cell(0, 6, f"Page {page + 1}", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
