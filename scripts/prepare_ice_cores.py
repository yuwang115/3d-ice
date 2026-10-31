#!/usr/bin/env python3
"""Build the ice-core site catalogues for the explorer's Places & Geographic Features.

The source of record is ``scripts/data/ice_core_sites.json``: a hand-checked table of the
deep and representative ice-core drilling sites of Antarctica and Greenland. No single
archive lists them cleanly (NOAA NCEI holds composites, marine cores and duplicate site
names; the 2022 Zenodo compilation of archived cores has sign errors), so every site gives
its position from at least two independent sources, and this script refuses a site whose
sources disagree by more than ``MAX_SOURCE_DISAGREEMENT_KM``. A refresh is therefore an
explicit, reviewable edit of that table, not a download.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

from prepare_polar_features import catalogue, normalize_text, project_polar, slugify


SITE_TABLE = Path(__file__).resolve().parent / "data" / "ice_core_sites.json"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "static" / "tools" / "data"

REGIONS = {
    "antarctica": {"epsg": 3031, "projection": "EPSG:3031", "filename": "antarctica_ice_cores.json"},
    "greenland": {"epsg": 3413, "projection": "EPSG:3413", "filename": "greenland_ice_cores.json"},
}
# Label and display priority of each kind of core: deep cores are labelled first.
CORE_TYPES = {
    "deep": ("Deep ice core", 1),
    "intermediate": ("Intermediate-depth ice core", 2),
    "blue_ice": ("Blue-ice drilling site", 2),
    "shallow": ("Shallow ice core", 3),
}
REQUIRED_FIELDS = ("name", "region", "core_type", "project", "operator", "lat", "lon", "reference")
TEXT_FIELDS = ("name", "project", "operator")
# The explorer drops any record whose name or id is longer than this.
MAX_TEXT_LENGTH = 180
MAX_SOURCE_DISAGREEMENT_KM = 5.0
MAX_DEPTH_M = 4000.0
ELEVATION_RANGE_M = (-100.0, 4200.0)
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
DRILLED_PATTERN = re.compile(r"^\d{4}(–\d{4})?$")
# Lets a visitor of the Chinese page find every site by typing "ice core".
SEARCH_ALIAS_ZH = "冰芯"
CATALOGUE_SOURCES = [
    {
        "name": "3D ICE compilation of ice-core sites (scripts/data/ice_core_sites.json)",
        "url": "https://github.com/yuwang115/3d-ice/blob/main/scripts/data/ice_core_sites.json",
        "attribution": "Positions checked against two sources per site; each site links the publication of its core.",
        "license": "Facts compiled from the sources below; cite the linked publications.",
    },
    {
        "name": "NOAA NCEI Paleoclimatology ice-core archive",
        "url": "https://www.ncei.noaa.gov/products/paleoclimatology/ice-core",
        "attribution": "NOAA National Centers for Environmental Information",
        "license": "U.S. Government work; cite the original publications.",
    },
    {
        "name": "Global ice drilling and archive location data for select ice cores",
        "url": "https://doi.org/10.5281/zenodo.7076469",
        "attribution": "Davidge, L., Brooks, H. L. and Mah, M. L. (2022), Zenodo",
        "license": "CC BY 4.0",
    },
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on a sphere of the Earth's mean radius."""

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _in_region(region: str, lat: float, lon: float) -> bool:
    if region == "antarctica":
        return -90 <= lat <= -60 and -180 <= lon <= 180
    return 59 <= lat <= 84 and -75 <= lon <= -10


def _check_optional_number(site: dict[str, Any], field: str, low: float, high: float) -> list[str]:
    value = site.get(field)
    if value is None:
        return []
    if not _finite(value) or not low <= value <= high:
        return [f"{field} must be null or a number in [{low}, {high}]"]
    return []


def site_errors(site: dict[str, Any]) -> list[str]:
    """Every reason a site cannot be published; empty when it can."""

    missing = [field for field in REQUIRED_FIELDS if site.get(field) in (None, "", [])]
    if missing:
        return [f"missing {', '.join(missing)}"]
    errors: list[str] = []
    region = site["region"]
    lat, lon = site["lat"], site["lon"]
    if region not in REGIONS:
        errors.append(f"unknown region {region!r}")
    if site["core_type"] not in CORE_TYPES:
        errors.append(f"unknown core_type {site['core_type']!r}")
    if not (_finite(lat) and _finite(lon)) or (region in REGIONS and not _in_region(region, lat, lon)):
        errors.append(f"position {lat}, {lon} is not in {region}")
        return errors
    for field in TEXT_FIELDS:
        value = site[field]
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_TEXT_LENGTH:
            errors.append(f"{field} must be text of 1–{MAX_TEXT_LENGTH} characters")
    # slugify falls back to "feature", so two such names would share an id.
    if isinstance(site["name"], str) and not normalize_text(site["name"]):
        errors.append("name needs Latin letters or digits for its id")
    aliases = site.get("aliases")
    if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
        errors.append("aliases must be a list of strings")
    errors += _drilled_errors(site.get("drilled"))
    errors += _check_optional_number(site, "depth_m", 1.0, MAX_DEPTH_M)
    errors += _check_optional_number(site, "oldest_ka", 0.001, 10_000.0)
    errors += _check_optional_number(site, "elevation_m", *ELEVATION_RANGE_M)
    if site.get("reached_bed") is not None and not isinstance(site["reached_bed"], bool):
        errors.append("reached_bed must be true, false or null")
    if site.get("reached_bed") is True and site.get("depth_m") is None:
        errors.append("a core that reached the bed needs its depth")
    doi = str(site["reference"].get("doi") or "") if isinstance(site["reference"], dict) else ""
    citation = str(site["reference"].get("citation") or "").strip() if isinstance(site["reference"], dict) else ""
    if not DOI_PATTERN.match(doi) or not citation:
        errors.append("reference needs a citation and a DOI")
    errors += coordinate_source_errors(site)
    return errors


def _drilled_errors(drilled: Any) -> list[str]:
    if drilled is None:
        return []
    match = DRILLED_PATTERN.fullmatch(drilled) if isinstance(drilled, str) else None
    years = [int(year) for year in re.findall(r"\d{4}", drilled)] if match else []
    if not match or years != sorted(years) or not 1900 <= years[0] <= years[-1] <= date.today().year:
        return [f"drilled {drilled!r} must be a year or a rising year range with an en dash"]
    return []


def coordinate_source_errors(site: dict[str, Any]) -> list[str]:
    """Two independent sources must place the site within MAX_SOURCE_DISAGREEMENT_KM of it."""

    sources = site.get("coordinate_sources") or []
    checked = [
        source for source in sources
        if isinstance(source, dict)
        and str(source.get("url", "")).startswith("https://")
        and _finite(source.get("lat"))
        and _finite(source.get("lon"))
    ]
    # A fragment names a place in the same document, not another source.
    if len({urllib.parse.urldefrag(source["url"]).url for source in checked}) < 2:
        return ["needs two coordinate sources with distinct https URLs"]
    errors = []
    for source in checked:
        distance = haversine_km(site["lat"], site["lon"], source["lat"], source["lon"])
        if distance > MAX_SOURCE_DISAGREEMENT_KM:
            errors.append(f"{source['url']} places it {distance:.1f} km away")
    return errors


def _alias_key(value: str) -> str:
    # normalize_text keeps only Latin letters and digits; keep the Chinese term as it is.
    return normalize_text(value) or value.strip()


def _aliases(site: dict[str, Any]) -> list[str]:
    seen = {_alias_key(site["name"])}
    aliases = []
    for alias in [*site.get("aliases", []), SEARCH_ALIAS_ZH]:
        key = _alias_key(alias)
        if key and key not in seen:
            seen.add(key)
            aliases.append(alias.strip())
    return aliases


def build_item(site: dict[str, Any]) -> dict[str, Any]:
    """The runtime record of one site, projected onto its region's polar stereographic grid."""

    region = site["region"]
    feature_type, priority = CORE_TYPES[site["core_type"]]
    # Five decimals is about a metre; project the rounded position so the record agrees with itself.
    lat, lon = round(float(site["lat"]), 5), round(float(site["lon"]), 5)
    x_m, y_m = project_polar(lat, lon, REGIONS[region]["epsg"])
    doi = site["reference"]["doi"]
    return {
        "id": f"{region}-ice-core-{slugify(site['name'])}",
        "region": region,
        "layer": "ice_cores",
        "kind": "ice_core",
        "name": site["name"].strip(),
        "name_zh": "",
        "aliases": _aliases(site),
        "core_type": site["core_type"],
        "feature_type": feature_type,
        "project": site["project"].strip(),
        "operator": site["operator"].strip(),
        "drilled": site.get("drilled") or "",
        "depth_m": site.get("depth_m"),
        "reached_bed": site.get("reached_bed"),
        "oldest_ka": site.get("oldest_ka"),
        "elevation_m": site.get("elevation_m"),
        "display_priority": priority,
        "lat": lat,
        "lon": lon,
        "x_m": x_m,
        "y_m": y_m,
        "citation": site["reference"]["citation"].strip(),
        "source_url": f"https://doi.org/{urllib.parse.quote(doi, safe='/()._-;:')}",
    }


def build_catalogues(sites: list[dict[str, Any]], as_of: str) -> dict[str, dict[str, Any]]:
    problems = [(site.get("name") or f"site {index}", site_errors(site)) for index, site in enumerate(sites)]
    problems = [(name, errors) for name, errors in problems if errors]
    if problems:
        detail = "; ".join(f"{name}: {', '.join(errors)}" for name, errors in problems)
        raise ValueError(f"Ice-core sites failed validation: {detail}")
    items = [build_item(site) for site in sites]
    ids = [item["id"] for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate ice-core site ids: {', '.join(duplicates)}")
    payloads = {}
    for region, config in REGIONS.items():
        regional = sorted(
            (item for item in items if item["region"] == region),
            key=lambda item: (item["display_priority"], item["name"]),
        )
        payloads[config["filename"]] = catalogue(
            region=region,
            layer="ice_cores",
            projection=config["projection"],
            as_of=as_of,
            sources=CATALOGUE_SOURCES,
            items=regional,
        )
    return payloads


def load_site_table(path: Path) -> dict[str, Any]:
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("schema_version") != 1 or not isinstance(table.get("sites"), list) or not table.get("as_of"):
        raise ValueError(f"{path} is not a version-1 ice-core site table")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=SITE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--as-of", default=None, help="Snapshot date; defaults to the table's as_of")
    args = parser.parse_args()
    table = load_site_table(args.sites)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in build_catalogues(table["sites"], args.as_of or table["as_of"]).items():
        destination = args.output_dir / filename
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {destination} ({payload['feature_count']} sites)")


if __name__ == "__main__":
    main()
