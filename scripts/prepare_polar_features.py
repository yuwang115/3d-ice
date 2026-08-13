#!/usr/bin/env python3
"""Build searchable polar station and geographic-name catalogues.

The generated JSON files are intentionally small, static snapshots.  They keep
the source URL and snapshot date beside every record so a future refresh is an
explicit, reviewable operation rather than a silent runtime dependency.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable


COMNAP_CSV_URL = "https://www.comnap.aq/s/Facilities_Nov2024.csv"
COMNAP_INFO_URL = "https://www.comnap.aq/antarctic-facilities-information"
INTERACT_API_URL = "https://www.interact-gis.org/api/stationlist"
INTERACT_INFO_URL = "https://www.interact-gis.org/"
SCAR_API_URL = "https://placenames.aq/api/place_names_consolidated"
SCAR_INFO_URL = "https://placenames.aq/information"
NUNAGIS_LAYER_URL = (
    "https://kort.nunagis.gl/refserver/rest/services/PlacenamesRegister/"
    "PlacenamesRegisterSearch/MapServer/1"
)
GEBCO_LAYER_ROOT = (
    "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
    "Undersea_Features/FeatureServer"
)
GEBCO_INFO_URL = "https://www.gebco.net/data-products/undersea-feature-names"
NATURAL_EARTH_INFO_URL = "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-physical-labels/"
ALLOWED_SOURCE_HOSTS = {
    "www.comnap.aq",
    "static1.squarespace.com",
    "www.interact-gis.org",
    "placenames.aq",
    "kort.nunagis.gl",
    "services2.arcgis.com",
}
MAX_RESPONSE_BYTES = 25 * 1024 * 1024
MAX_PAGE_COUNT = 100
MAX_FEATURE_COUNT = 100_000

SCAR_FEATURE_TYPES = {
    293: "sea",
    298: "ice_shelf",
    354: "basin",
    405: "plateau",
    411: "mountain_range",
    426: "basin",
    427: "mountain_range",
}

GREENLAND_FEATURE_TYPES = {
    44: "mountain_range",
    46: "mountain_range",
    49: "mountain",
    50: "plateau",
    54: "mountain",
    57: "fjord",
    88: "ice_sheet",
    117: "nunatak",
    166: "strait",
    178: "sea",
    186: "ocean",
}

GEOGRAPHIC_NAME_QUOTAS = {
    "antarctica": {
        "ocean": 1,
        "sea": 14,
        "basin": 28,
        "mountain_range": 64,
        "plateau": 28,
        "ice_shelf": 25,
    },
    "greenland": {
        "ocean": 4,
        "sea": 18,
        "basin": 6,
        "mountain_range": 32,
        "mountain": 21,
        "plateau": 18,
        "fjord": 26,
        "strait": 20,
        "ice_sheet": 1,
        "nunatak": 14,
    },
}

GREENLAND_STATION_IDS = {72, 73, 74, 75, 76, 77, 78, 90, 95}

COUNTRY_ISO3 = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Chile": "CHL",
    "China": "CHN",
    "Czech Republic": "CZE",
    "Ecuador": "ECU",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greenland": "GRL",
    "India": "IND",
    "Italy": "ITA",
    "Japan": "JPN",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Peru": "PER",
    "Poland": "POL",
    "Republic of Belarus": "BLR",
    "Republic of Korea": "KOR",
    "Russia": "RUS",
    "South Africa": "ZAF",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
}

STATION_NAME_ZH = {
    "amundsen scott south pole station": "阿蒙森-斯科特南极站",
    "mcmurdo station": "麦克默多站",
    "great wall": "长城站",
    "zhongshan station": "中山站",
    "qinling": "秦岭站",
    "kunlun station": "昆仑站",
    "taishan camp": "泰山站",
    "rothera research station": "罗瑟拉站",
    "halley vi research station": "哈雷六号站",
    "vostok": "沃斯托克站",
    "concordia": "康科迪亚站",
    "summit station": "格陵兰峰顶站",
    "zackenberg research station": "扎肯伯格科考站",
    "villum research station vrs": "维卢姆科考站",
    "arctic station": "北极站",
    "sermilik research station": "塞米利克科考站",
}

PLACE_NAME_ZH = {
    "southern ocean": "南大洋",
    "ross sea": "罗斯海",
    "weddell sea": "威德尔海",
    "amundsen sea": "阿蒙森海",
    "bellingshausen sea": "别林斯高晋海",
    "transantarctic mountains": "横贯南极山脉",
    "ellsworth mountains": "埃尔斯沃思山脉",
    "south polar plateau": "南极高原",
    "wilkes subglacial basin": "威尔克斯冰下盆地",
    "bentley subglacial trench": "本特利冰下沟谷",
    "greenland sea": "格陵兰海",
    "baffin bay": "巴芬湾",
    "davis strait": "戴维斯海峡",
    "denmark strait": "丹麦海峡",
    "labrador sea": "拉布拉多海",
    "kane basin": "凯恩盆地",
    "labrador basin": "拉布拉多盆地",
    "irminger basin": "伊尔明厄盆地",
    "gunnbjorn fjeld": "贡比约恩山",
    "watkins bjerge": "沃特金斯山脉",
    "stauning alper": "斯陶宁阿尔卑斯山脉",
    "kangertittivaq": "康格蒂蒂瓦克峡湾",
    "qeqertarsuup tunua": "迪斯科湾",
}

GREENLAND_ENGLISH_ALIASES = {
    13358: ["Davis Strait"],
    35623: ["Baffin Bay"],
    13424: ["Labrador Sea"],
    13363: ["Denmark Strait"],
    13369: ["Greenland Sea"],
    10128: ["Disko Bay"],
    31041: ["Kane Basin"],
    24739: ["Gunnbjørn Fjeld", "Greenland's highest mountain"],
    24750: ["Watkins Range"],
    25654: ["Stauning Alps"],
    26229: ["Stauning Alps"],
    25129: ["Scoresby Sound"],
}

KEY_PLACE_NAMES = {
    "southern ocean",
    "ross sea",
    "weddell sea",
    "amundsen sea",
    "bellingshausen sea",
    "transantarctic mountains",
    "ellsworth mountains",
    "south polar plateau",
    "wilkes subglacial basin",
    "ross ice shelf",
    "greenland sea",
    "baffin bay",
    "davis strait",
    "denmark strait",
    "labrador sea",
    "kane basin",
    "gunnbjorn fjeld",
    "watkins bjerge",
    "stauning alper",
    "kangertittivaq",
    "qeqertarsuup tunua",
    "greenland ice sheet",
}

GREENLAND_OPERATOR_OVERRIDES = {
    72: "University of Copenhagen",
    73: "Greenland Institute of Natural Resources",
    74: "University of Copenhagen",
    75: "U.S. National Science Foundation",
    76: "University of Copenhagen / international EGRIP consortium",
    77: "Aarhus University",
    78: "Aarhus University",
    90: "Danish Meteorological Institute",
    95: "Technical University of Denmark",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.translate(
        str.maketrans(
            {
                "ø": "o",
                "Ø": "o",
                "ł": "l",
                "Ł": "l",
                "ð": "d",
                "Ð": "d",
                "þ": "th",
                "Þ": "th",
                "æ": "ae",
                "Æ": "ae",
                "œ": "oe",
                "Œ": "oe",
            }
        )
    )
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "" if text in {"", "-"} else text


def unique_text(values: Iterable[Any], *, exclude: str = "") -> list[str]:
    result: list[str] = []
    seen = {normalize_text(exclude)} if exclude else set()
    for value in values:
        text = clean_text(value)
        key = normalize_text(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def slugify(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(value)).strip("-")
    return slug or "feature"


def longitude_sector(longitude: float, sector_count: int = 12) -> int:
    """Return a stable global longitude sector for spatial sampling."""

    if sector_count < 1:
        raise ValueError("sector_count must be positive")
    normalized = (float(longitude) + 180.0) % 360.0
    return min(sector_count - 1, int(normalized / (360.0 / sector_count)))


def spatial_cell(latitude: float, longitude: float) -> tuple[int, int]:
    """Return a 5-degree latitude band and 30-degree longitude sector."""

    lat = float(latitude)
    lon = float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError("Invalid feature coordinate")
    latitude_band = min(35, int((lat + 90.0) / 5.0))
    return latitude_band, longitude_sector(lon)


def feature_name_terms(item: dict[str, Any]) -> set[str]:
    return {
        normalized
        for normalized in (normalize_text(value) for value in [item.get("name"), *(item.get("aliases") or [])])
        if normalized
    }


def feature_source_rank(item: dict[str, Any]) -> int:
    """Prefer official regional gazetteers over overlapping global records."""

    item_id = str(item.get("id", ""))
    return 1 if "-gebco-" in item_id else 0


def deduplicate_semantic_features(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    accepted_terms: dict[str, list[set[str]]] = defaultdict(list)
    ordered = sorted(
        items,
        key=lambda item: (
            int(item.get("display_priority", 99)),
            feature_source_rank(item),
            normalize_text(item.get("name")),
            str(item.get("id", "")),
        ),
    )
    for item in ordered:
        kind = str(item.get("kind", ""))
        terms = feature_name_terms(item)
        if any(terms & prior_terms for prior_terms in accepted_terms[kind]):
            continue
        accepted.append(item)
        accepted_terms[kind].append(terms)
    return accepted


def select_balanced_features(
    items: Iterable[dict[str, Any]],
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    """Select key names by kind, priority, and spatial-cell diversity.

    Priority-one records are curated landmarks and are always retained. The
    remainder is selected round-robin from latitude/longitude cells within each
    priority band so deterministic snapshots cover the mapped region instead
    of favouring alphabetically early names.
    """

    candidates = [item for item in items]
    if not quotas or any(not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0 for quota in quotas.values()):
        raise ValueError("Feature quotas must be positive integers")
    candidate_ids: set[str] = set()
    for item in candidates:
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            raise ValueError("Every feature must have a non-empty id")
        if item_id in candidate_ids:
            raise ValueError(f"Found duplicate feature id: {item_id}")
        candidate_ids.add(item_id)
        if not str(item.get("name", "")).strip():
            raise ValueError(f"Feature {item_id} must have a name")
        spatial_cell(float(item.get("lat", math.nan)), float(item.get("lon", math.nan)))
    unconfigured_kinds = {str(item.get("kind", "")) for item in candidates} - set(quotas)
    if unconfigured_kinds:
        raise ValueError(f"Found unconfigured feature kinds: {', '.join(sorted(unconfigured_kinds))}")

    candidates = deduplicate_semantic_features(candidates)
    selected: list[dict[str, Any]] = []
    kind_order = {kind: index for index, kind in enumerate(quotas)}

    for kind, raw_quota in quotas.items():
        quota = raw_quota
        kind_items = [item for item in candidates if item.get("kind") == kind]
        mandatory = sorted(
            (item for item in kind_items if int(item.get("display_priority", 99)) == 1),
            key=lambda item: (normalize_text(item.get("name")), str(item.get("id", ""))),
        )
        if len(mandatory) > quota:
            raise ValueError(f"The {kind} quota cannot retain all priority-one features")

        kind_selected = list(mandatory)
        selected_ids = {str(item.get("id", "")) for item in kind_selected}
        selected_cells = {spatial_cell(float(item["lat"]), float(item["lon"])) for item in kind_selected}
        remaining = [item for item in kind_items if str(item.get("id", "")) not in selected_ids]

        for priority in sorted({int(item.get("display_priority", 99)) for item in remaining}):
            if len(kind_selected) >= quota:
                break
            buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for item in remaining:
                if int(item.get("display_priority", 99)) != priority:
                    continue
                buckets[spatial_cell(float(item["lat"]), float(item["lon"]))].append(item)
            for bucket in buckets.values():
                bucket.sort(key=lambda item: (normalize_text(item.get("name")), str(item.get("id", ""))))

            while buckets and len(kind_selected) < quota:
                cell_order = sorted(buckets, key=lambda cell: (cell in selected_cells, cell))
                for cell in cell_order:
                    if len(kind_selected) >= quota:
                        break
                    bucket = buckets.get(cell, [])
                    if not bucket:
                        buckets.pop(cell, None)
                        continue
                    kind_selected.append(bucket.pop(0))
                    selected_cells.add(cell)
                    if not bucket:
                        buckets.pop(cell, None)

        if len(kind_selected) != quota:
            raise ValueError(f"Only {len(kind_selected)} of {quota} requested {kind} features are available")
        selected.extend(kind_selected)

    return sorted(
        selected,
        key=lambda item: (
            int(item.get("display_priority", 99)),
            kind_order.get(str(item.get("kind")), len(kind_order)),
            normalize_text(item.get("name")),
            str(item.get("id", "")),
        ),
    )


def project_polar(lat: float, lon: float, epsg: int) -> tuple[float, float]:
    """Project WGS84 lon/lat to EPSG:3031 or EPSG:3413."""

    if epsg not in {3031, 3413}:
        raise ValueError(f"Unsupported polar projection: EPSG:{epsg}")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinate: {lat}, {lon}")

    semi_major = 6_378_137.0
    eccentricity = math.sqrt(0.0066943799901413165)
    north = epsg == 3413
    latitude = math.radians(abs(lat))
    standard_parallel = math.radians(70 if north else 71)
    central_meridian = math.radians(-45 if north else 0)
    longitude = math.radians(lon)

    def t_value(phi: float) -> float:
        sin_phi = math.sin(phi)
        correction = ((1 - eccentricity * sin_phi) / (1 + eccentricity * sin_phi)) ** (eccentricity / 2)
        return math.tan(math.pi / 4 - phi / 2) / correction

    sin_standard = math.sin(standard_parallel)
    m_standard = math.cos(standard_parallel) / math.sqrt(1 - eccentricity**2 * sin_standard**2)
    radius = semi_major * m_standard * t_value(latitude) / t_value(standard_parallel)
    delta = longitude - central_meridian
    x_m = radius * math.sin(delta)
    y_m = (-1 if north else 1) * radius * math.cos(delta)
    return round(x_m, 3), round(y_m, 3)


def fetch_bytes(url: str, params: dict[str, Any] | None = None) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    requested_host = urllib.parse.urlparse(url).hostname
    if requested_host not in ALLOWED_SOURCE_HOSTS:
        raise ValueError(f"Unexpected polar data host: {requested_host}")
    request = urllib.request.Request(url, headers={"User-Agent": "3D-ICE-data-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        final_host = urllib.parse.urlparse(response.geturl()).hostname
        if final_host not in ALLOWED_SOURCE_HOSTS:
            raise ValueError(f"Unexpected redirect host: {final_host}")
        declared_size = int(response.headers.get("Content-Length") or 0)
        if declared_size > MAX_RESPONSE_BYTES:
            raise ValueError(f"Polar data response is too large: {declared_size} bytes")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError(f"Polar data response exceeded {MAX_RESPONSE_BYTES} bytes")
        return payload


def fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    return json.loads(fetch_bytes(url, params).decode("utf-8"))


def add_projected_position(item: dict[str, Any], epsg: int) -> dict[str, Any]:
    lat = float(item["lat"])
    lon = float(item["lon"])
    x_m, y_m = project_polar(lat, lon, epsg)
    return {**item, "lat": lat, "lon": lon, "x_m": x_m, "y_m": y_m}


def name_zh_for(name: str, aliases: Iterable[str], mapping: dict[str, str]) -> str:
    for candidate in (name, *aliases):
        translated = mapping.get(normalize_text(candidate))
        if translated:
            return translated
    return ""


def fetch_antarctic_stations() -> list[dict[str, Any]]:
    text = fetch_bytes(COMNAP_CSV_URL).decode("latin-1")
    rows = csv.DictReader(io.StringIO(text))
    required_fields = {"Record ID#", "Type", "Official Name", "Latitude (DD)", "Longitude (DD)"}
    if not required_fields.issubset(set(rows.fieldnames or [])):
        raise ValueError("COMNAP returned an unexpected CSV schema")
    items: list[dict[str, Any]] = []
    for row in rows:
        if clean_text(row.get("Type")) != "Station":
            continue
        lat = float(row["Latitude (DD)"])
        if lat > -60:
            continue
        lon = float(row["Longitude (DD)"])
        official_name = clean_text(row.get("Official Name"))
        english_name = clean_text(row.get("English Name"))
        name = official_name or english_name
        aliases = unique_text([english_name], exclude=name)
        country = clean_text(row.get("Operator (primary)"))
        item = {
            "id": (
                f"antarctica-station-{slugify(name)}-"
                f"{clean_text(row.get('Record ID#'))}"
            ),
            "region": "antarctica",
            "layer": "research_stations",
            "kind": "research_station",
            "name": name,
            "name_zh": name_zh_for(name, aliases, STATION_NAME_ZH),
            "aliases": aliases,
            "operator": country,
            "additional_operator": clean_text(row.get("Operator (additional)")),
            "country": country,
            "country_iso3": COUNTRY_ISO3.get(country, ""),
            "status": clean_text(row.get("Status")).casefold().replace(" ", "_"),
            "seasonality": clean_text(row.get("Seasonality")).casefold().replace("-", "_").replace(" ", "_"),
            "year_established": clean_text(row.get("Year Established")),
            "elevation_m": clean_text(row.get("Elevation (meters)")),
            "display_priority": 1 if clean_text(row.get("Seasonality")) == "Year-Round" else 2,
            "lat": lat,
            "lon": lon,
            "source_url": COMNAP_INFO_URL,
        }
        items.append(add_projected_position(item, 3031))
    return sorted(items, key=lambda item: (item["country"], item["name"]))


def fetch_greenland_stations() -> list[dict[str, Any]]:
    stations = fetch_json(INTERACT_API_URL)
    if not isinstance(stations, list):
        raise ValueError("INTERACT returned an unexpected response shape")
    items: list[dict[str, Any]] = []
    for station in stations:
        station_id = int(station.get("StationId") or 0)
        if station_id not in GREENLAND_STATION_IDS:
            continue
        name = clean_text(station.get("StationName"))
        aliases = unique_text([station.get("Acronym")], exclude=name)
        country = clean_text(station.get("OperatingCountry"))
        status = "relocated" if station_id == 76 else "open"
        item = {
            "id": f"greenland-station-{slugify(name)}-{station_id}",
            "region": "greenland",
            "layer": "research_stations",
            "kind": "research_station",
            "name": name,
            "name_zh": name_zh_for(name, aliases, STATION_NAME_ZH),
            "aliases": aliases,
            "operator": GREENLAND_OPERATOR_OVERRIDES.get(station_id, country),
            "country": country,
            "country_iso3": COUNTRY_ISO3.get(country, ""),
            "status": status,
            "status_note": "Original EGRIP site cleared in 2025; programme relocated to GRIP/NewGRIP." if station_id == 76 else "",
            "seasonality": clean_text(station.get("OperatingPeriod")),
            "website": clean_text(station.get("Website")),
            "display_priority": 1 if "year-round" in clean_text(station.get("OperatingPeriod")).casefold() else 2,
            "lat": float(station["Latitude"]),
            "lon": float(station["Longitude"]),
            "source_url": INTERACT_INFO_URL,
        }
        items.append(add_projected_position(item, 3413))
    return sorted(items, key=lambda item: item["name"])


def fetch_scar_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_fields = (
        "name_id,place_id,place_name_mapping,place_name_gazetteer,latitude,longitude,"
        "gazetteer_code,gazetteer_name,feature_type_code,feature_type_name"
    )
    for feature_code in SCAR_FEATURE_TYPES:
        offset = 0
        for _page_number in range(MAX_PAGE_COUNT):
            page = fetch_json(
                SCAR_API_URL,
                {
                    "select": selected_fields,
                    "feature_type_code": f"eq.{feature_code}",
                    "limit": 1000,
                    "offset": offset,
                    "order": "place_id.asc,name_id.asc",
                },
            )
            if not isinstance(page, list):
                raise ValueError("SCAR returned an unexpected response shape")
            rows.extend(page)
            if len(rows) > MAX_FEATURE_COUNT:
                raise ValueError("SCAR feature limit exceeded")
            if len(page) < 1000:
                break
            offset += len(page)
        else:
            raise ValueError("SCAR page limit exceeded")
    return rows


def fetch_antarctic_names() -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in fetch_scar_rows():
        grouped[int(row["place_id"])].append(row)

    items: list[dict[str, Any]] = []
    for place_id, rows in grouped.items():
        primary = next((row for row in rows if row.get("place_name_mapping")), rows[0])
        lat = float(primary["latitude"])
        lon = float(primary["longitude"])
        name = clean_text(primary.get("place_name_mapping") or primary.get("place_name_gazetteer"))
        aliases = unique_text(
            [row.get("place_name_mapping") for row in rows] + [row.get("place_name_gazetteer") for row in rows],
            exclude=name,
        )
        kind = SCAR_FEATURE_TYPES[int(primary["feature_type_code"])]
        normalized_candidates = {normalize_text(value) for value in [name, *aliases]}
        priority = 1 if normalized_candidates & KEY_PLACE_NAMES else 2 if kind in {"sea", "basin"} else 3
        item = {
            "id": f"antarctica-place-scar-{place_id}",
            "source_id": place_id,
            "region": "antarctica",
            "layer": "geographic_names",
            "kind": kind,
            "feature_type": clean_text(primary.get("feature_type_name")),
            "name": name,
            "name_zh": name_zh_for(name, aliases, PLACE_NAME_ZH),
            "aliases": aliases,
            "gazetteers": unique_text(row.get("gazetteer_name") for row in rows),
            "display_priority": priority,
            "lat": lat,
            "lon": lon,
            "source_url": SCAR_INFO_URL,
        }
        items.append(add_projected_position(item, 3031))

    if not any("ellsworth mountains" in feature_name_terms(item) for item in items):
        items.append(
            add_projected_position(
                {
                    "id": "antarctica-place-scar-4205",
                    "source_id": 4205,
                    "region": "antarctica",
                    "layer": "geographic_names",
                    "kind": "mountain_range",
                    "feature_type": "Mountain range",
                    "name": "Ellsworth Mountains",
                    "name_zh": "埃尔斯沃思山脉",
                    "aliases": [],
                    "gazetteers": [],
                    "display_priority": 1,
                    "lat": -79.0,
                    "lon": -84.5,
                    "source_url": SCAR_INFO_URL,
                },
                3031,
            )
        )

    # SCAR covers features south of 60 S but does not currently carry a single
    # entry for the surrounding ocean at the scale used by this explorer.
    items.append(
        add_projected_position(
            {
                "id": "antarctica-place-natural-earth-southern-ocean",
                "source_id": "southern-ocean",
                "region": "antarctica",
                "layer": "geographic_names",
                "kind": "ocean",
                "feature_type": "Ocean",
                "name": "Southern Ocean",
                "name_zh": "南大洋",
                "aliases": ["Antarctic Ocean"],
                "display_priority": 1,
                "lat": -63.0,
                "lon": 0.0,
                "source_url": NATURAL_EARTH_INFO_URL,
            },
            3031,
        )
    )
    return sorted(items, key=lambda item: (item["display_priority"], item["name"]))


def fetch_arcgis_features(layer_url: str, where: str, out_fields: str) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    offset = 0
    for _page_number in range(MAX_PAGE_COUNT):
        page = fetch_json(
            f"{layer_url}/query",
            {
                "where": where,
                "outFields": out_fields,
                "returnGeometry": "true",
                "outSR": 4326,
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "orderByFields": "OBJECTID ASC",
                "f": "json",
            },
        )
        if not isinstance(page, dict) or not isinstance(page.get("features"), list):
            raise ValueError(f"ArcGIS returned an unexpected response shape: {page.get('error') if isinstance(page, dict) else type(page).__name__}")
        batch = page.get("features", [])
        features.extend(batch)
        if len(features) > MAX_FEATURE_COUNT:
            raise ValueError("ArcGIS feature limit exceeded")
        if len(batch) < 2000 or not page.get("exceededTransferLimit"):
            break
        offset += len(batch)
    else:
        raise ValueError("ArcGIS page limit exceeded")
    return features


def fetch_greenland_official_names() -> list[dict[str, Any]]:
    type_codes = ",".join(str(code) for code in GREENLAND_FEATURE_TYPES)
    features = fetch_arcgis_features(
        NUNAGIS_LAYER_URL,
        f"Type IN ({type_codes})",
        (
            "OBJECTID,ID,PlacenameOfficial,PlacenameOfficialOld,PlacenameVariant,PlacenameDanish,"
            "PlacenameInternational,PlacenameAlternative,Category,SubCategory,Type,EditDate"
        ),
    )
    items: list[dict[str, Any]] = []
    for feature in features:
        attributes = feature.get("attributes", {})
        geometry = feature.get("geometry", {})
        if not isinstance(geometry.get("x"), (int, float)) or not isinstance(geometry.get("y"), (int, float)):
            continue
        source_id = int(attributes["ID"])
        name_candidates = [
            attributes.get("PlacenameOfficial"),
            attributes.get("PlacenameInternational"),
            attributes.get("PlacenameDanish"),
            attributes.get("PlacenameVariant"),
            attributes.get("PlacenameOfficialOld"),
            attributes.get("PlacenameAlternative"),
            *GREENLAND_ENGLISH_ALIASES.get(source_id, []),
        ]
        name = next((clean_text(candidate) for candidate in name_candidates if clean_text(candidate)), "")
        if not name:
            continue
        aliases = unique_text(
            name_candidates,
            exclude=name,
        )
        kind = "basin" if source_id == 31041 else GREENLAND_FEATURE_TYPES[int(attributes["Type"])]
        normalized_candidates = {normalize_text(value) for value in [name, *aliases]}
        priority = 1 if normalized_candidates & KEY_PLACE_NAMES else 2 if kind in {"ocean", "mountain_range", "plateau"} else 4
        item = {
            "id": f"greenland-place-nunagis-{source_id}",
            "source_id": source_id,
            "region": "greenland",
            "layer": "geographic_names",
            "kind": kind,
            "feature_type": str(attributes.get("Type")),
            "name": name,
            "name_zh": name_zh_for(name, aliases, PLACE_NAME_ZH),
            "aliases": aliases,
            "display_priority": priority,
            "lat": float(geometry["y"]),
            "lon": float(geometry["x"]),
            "source_url": f"{NUNAGIS_LAYER_URL}",
        }
        items.append(add_projected_position(item, 3413))
    return items


def geometry_centroid(geometry: dict[str, Any]) -> tuple[float, float] | None:
    points = geometry.get("points")
    if points:
        return float(points[0][0]), float(points[0][1])
    paths = geometry.get("paths")
    if paths and paths[0]:
        flat = paths[0]
        return sum(point[0] for point in flat) / len(flat), sum(point[1] for point in flat) / len(flat)
    rings = geometry.get("rings")
    if rings and rings[0]:
        flat = rings[0]
        return sum(point[0] for point in flat) / len(flat), sum(point[1] for point in flat) / len(flat)
    return None


def fetch_greenland_undersea_basins() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for layer_id in (0, 1, 2):
        features = fetch_arcgis_features(
            f"{GEBCO_LAYER_ROOT}/{layer_id}",
            "TYPE = 'Basin'",
            "OBJECTID,FEATURE_ID,NAME,TYPE",
        )
        for feature in features:
            centroid = geometry_centroid(feature.get("geometry", {}))
            if not centroid:
                continue
            lon, lat = centroid
            if not (-75 <= lon <= -5 and 55 <= lat <= 85):
                continue
            attributes = feature["attributes"]
            name = f"{clean_text(attributes.get('NAME'))} Basin"
            source_id = int(attributes["FEATURE_ID"])
            item = {
                "id": f"greenland-place-gebco-{source_id}",
                "source_id": source_id,
                "region": "greenland",
                "layer": "geographic_names",
                "kind": "basin",
                "feature_type": "Undersea basin",
                "name": name,
                "name_zh": name_zh_for(name, [], PLACE_NAME_ZH),
                "aliases": [],
                "display_priority": 1 if normalize_text(name) in KEY_PLACE_NAMES else 2,
                "lat": lat,
                "lon": lon,
                "source_url": GEBCO_INFO_URL,
            }
            items.append(add_projected_position(item, 3413))
    return items


def fetch_greenland_names() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in [*fetch_greenland_official_names(), *fetch_greenland_undersea_basins()]:
        by_id[item["id"]] = item
    greenland_ice_sheet = add_projected_position(
        {
            "id": "greenland-place-natural-earth-greenland-ice-sheet",
            "source_id": "greenland-ice-sheet",
            "region": "greenland",
            "layer": "geographic_names",
            "kind": "ice_sheet",
            "feature_type": "Ice sheet",
            "name": "Greenland Ice Sheet",
            "name_zh": "格陵兰冰盖",
            "aliases": ["Inland Ice", "Kalaallit Nunaata Sermersua"],
            "display_priority": 1,
            "lat": 72.0,
            "lon": -40.0,
            "source_url": NATURAL_EARTH_INFO_URL,
        },
        3413,
    )
    by_id[greenland_ice_sheet["id"]] = greenland_ice_sheet
    return sorted(by_id.values(), key=lambda item: (item["display_priority"], item["name"]))


def catalogue(
    *,
    region: str,
    layer: str,
    projection: str,
    as_of: str,
    sources: list[dict[str, str]],
    items: list[dict[str, Any]],
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "region": region,
        "layer": layer,
        "projection": projection,
        "as_of": as_of,
        "sources": sources,
        "feature_count": len(items),
        "items": items,
    }
    if selection:
        payload["selection"] = selection
    return payload


def build_catalogues(as_of: str) -> dict[str, dict[str, Any]]:
    antarctic_stations = fetch_antarctic_stations()
    greenland_stations = fetch_greenland_stations()
    antarctic_name_candidates = fetch_antarctic_names()
    greenland_name_candidates = fetch_greenland_names()
    antarctic_names = select_balanced_features(
        antarctic_name_candidates,
        GEOGRAPHIC_NAME_QUOTAS["antarctica"],
    )
    greenland_names = select_balanced_features(
        greenland_name_candidates,
        GEOGRAPHIC_NAME_QUOTAS["greenland"],
    )
    return {
        "antarctica_research_stations.json": catalogue(
            region="antarctica",
            layer="research_stations",
            projection="EPSG:3031",
            as_of=as_of,
            sources=[
                {
                    "name": "COMNAP Antarctic Facilities List (November 2024)",
                    "url": COMNAP_INFO_URL,
                    "attribution": "COMNAP 2025",
                    "license": "Educational and non-commercial use; contact COMNAP for other uses.",
                }
            ],
            items=antarctic_stations,
        ),
        "greenland_research_stations.json": catalogue(
            region="greenland",
            layer="research_stations",
            projection="EPSG:3413",
            as_of=as_of,
            sources=[
                {
                    "name": "INTERACT research-station directory",
                    "url": INTERACT_INFO_URL,
                    "attribution": "INTERACT GIS; status reviewed for the static 3D ICE snapshot",
                    "license": "Public directory; follow linked operator terms for reuse.",
                }
            ],
            items=greenland_stations,
        ),
        "antarctica_geographic_names.json": catalogue(
            region="antarctica",
            layer="geographic_names",
            projection="EPSG:3031",
            as_of=as_of,
            sources=[
                {
                    "name": "SCAR Composite Gazetteer of Antarctica",
                    "url": SCAR_INFO_URL,
                    "attribution": "Scientific Committee on Antarctic Research (SCAR)",
                    "license": "CC BY 4.0 via the Australian Antarctic Data Centre",
                },
                {
                    "name": "Natural Earth physical labels",
                    "url": NATURAL_EARTH_INFO_URL,
                    "attribution": "Natural Earth",
                    "license": "Public domain",
                },
            ],
            items=antarctic_names,
            selection={
                "strategy": "balanced_key_features",
                "limit": sum(GEOGRAPHIC_NAME_QUOTAS["antarctica"].values()),
                "source_feature_count": len(antarctic_name_candidates),
                "kind_quotas": GEOGRAPHIC_NAME_QUOTAS["antarctica"],
            },
        ),
        "greenland_geographic_names.json": catalogue(
            region="greenland",
            layer="geographic_names",
            projection="EPSG:3413",
            as_of=as_of,
            sources=[
                {
                    "name": "Greenland Place Names Register / NunaGIS",
                    "url": NUNAGIS_LAYER_URL,
                    "attribution": "Oqaasileriffik and the Greenland Place Names Board via NunaGIS",
                    "license": "Public GIS service; bulk redistribution terms are not stated by the service.",
                },
                {
                    "name": "IHO-IOC GEBCO Gazetteer of Undersea Feature Names",
                    "url": GEBCO_INFO_URL,
                    "attribution": "IHO-IOC GEBCO Gazetteer of Undersea Feature Names",
                    "license": "Attribution required; consult GEBCO for redistribution terms.",
                },
                {
                    "name": "Natural Earth physical labels",
                    "url": NATURAL_EARTH_INFO_URL,
                    "attribution": "Natural Earth",
                    "license": "Public domain",
                },
            ],
            items=greenland_names,
            selection={
                "strategy": "balanced_key_features",
                "limit": sum(GEOGRAPHIC_NAME_QUOTAS["greenland"].values()),
                "source_feature_count": len(greenland_name_candidates),
                "kind_quotas": GEOGRAPHIC_NAME_QUOTAS["greenland"],
            },
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "static" / "tools" / "data",
    )
    parser.add_argument("--as-of", default=date.today().isoformat())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, payload in build_catalogues(args.as_of).items():
        destination = args.output_dir / filename
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {destination} ({payload['feature_count']} features)")


if __name__ == "__main__":
    main()
