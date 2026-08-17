"""Consistency checks for public project and JOSS metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_json(name: str) -> dict:
    return json.loads((ROOT_DIR / name).read_text(encoding="utf-8"))


def load_citation() -> dict:
    return yaml.safe_load((ROOT_DIR / "CITATION.cff").read_text(encoding="utf-8"))


def load_paper_metadata() -> dict:
    paper = (ROOT_DIR / "paper.md").read_text(encoding="utf-8")
    _, front_matter, _ = paper.split("---", 2)
    return yaml.safe_load(front_matter)


def load_pyproject_version() -> str:
    pyproject = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_release_versions_are_consistent():
    citation = load_citation()
    codemeta = load_json("codemeta.json")
    package = load_json("package.json")

    assert citation["version"] == codemeta["version"]
    assert citation["version"] == package["version"]
    assert citation["version"] == load_pyproject_version()


def test_paper_and_citation_titles_are_consistent():
    assert load_paper_metadata()["title"] == load_citation()["title"]


def test_author_orcid_is_consistent():
    expected_orcid = "0000-0001-9070-6004"
    citation = load_citation()
    codemeta = load_json("codemeta.json")
    paper = load_paper_metadata()

    assert paper["authors"][0]["orcid"] == expected_orcid
    assert citation["authors"][0]["orcid"] == f"https://orcid.org/{expected_orcid}"
    assert codemeta["author"][0]["@id"] == f"https://orcid.org/{expected_orcid}"


def test_public_project_urls_are_consistent():
    citation = load_citation()
    codemeta = load_json("codemeta.json")

    assert citation["repository-code"] == codemeta["codeRepository"]
    assert citation["url"] == codemeta["url"] == "https://3d-ice.com/"
