"""Regression tests for source-data links on the introduction pages."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
HOME_PAGES = (
    ROOT_DIR / "static" / "index.html",
    ROOT_DIR / "static" / "zh" / "index.html",
)
BEDMAP3_URL = "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2"
QRF_URL = "https://doi.org/10.1017/jog.2025.10071"


@pytest.mark.integration
class TestHomeSourceData:
    def test_introduction_pages_link_all_new_terrain_sources(self):
        for page_path in HOME_PAGES:
            page = page_path.read_text(encoding="utf-8")

            assert 'id=source-data' in page
            assert BEDMAP3_URL in page
            assert QRF_URL in page
