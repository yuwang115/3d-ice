#!/usr/bin/env python3
"""Pack CPOM's Antarctic surface elevation change (radar altimetry, 1995-2026) for 3D ICE.

Inputs come from the CPOM Land Ice Portal (https://www.cpom.ucl.ac.uk/landice/), product
version 1.0, CC BY 4.0:

- the cumulative elevation change since March 1995, one file per 140-day epoch,
  `CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19950321-<end>-fv1.nc` (`dH`, `dH_uncertainty`);
- the 1995-2026 rate, `CPOM-AIS-L3C-DHDT-MULTIMISSION-5KM-199505-202601-fv1.nc`
  (`sec`, `sec_uncertainty`, `basin_id`);
- the ice-sheet and regional mass-change series, `get_timeseries.php?product=dmdt&basin=<id>`
  for AIS, WAIS, EAIS and APIS, saved as `dmdt_<id>.json`.

They are 5 km grids in EPSG:3031 whose rows run south to north, over grounded ice only.
Each explorer cell takes the area-weighted mean of the 5 km cells that overlap it, where
those cover at least half of it, and only where the explorer draws ice (grounded, floating
or Lake Vostok in BedMachine Antarctica v4). CPOM's mask is BedMachine v2, so values next to
the grounding line can fall on ice that v4 calls floating; they are kept, because the
strongest thinning is there.

Outputs:

- `cpom_antarctica_dhdt_{480,741}`: the rate map, a gridded overlay on the Balanced (10 km)
  and HD (4 km) terrain grids (`dhdt`, `dhdt_uncertainty`, `basin_id`);
- `cpom_antarctica_dh_480`: the cumulative change at every epoch, a sparse time series on
  the 10 km grid (`surface_change`, `final_uncertainty`, `domain_mask`), carrying CPOM's own
  mass-change series so the explorer quotes CPOM's totals, not sums over the resampled grid.

Two properties of the cumulative files shape the time series:

- The first epoch is not a zero field. The change is relative to CPOM's 1995 reference level,
  and the first 140-day epoch departs from it (median 9 cm, tens of metres at a few steep
  margins). The frames keep CPOM's values as published rather than subtracting that epoch.
- The per-epoch uncertainty is zero for about 78 % of cells before 2002, so it cannot say
  which early changes are significant. The package keeps only the uncertainty of the last
  epoch, which the explorer compares the change of every frame with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

INT16_FILL = -32768
INT16_MAX = 32767
CPOM_CELL_M = 5000.0
GT_PER_MM_SLE = 362.5
EXPLORER_ICE_FLAGS = (2, 3, 4)  # BedMachine Antarctica: grounded ice, floating ice, Lake Vostok
MIN_COVER = 0.5  # share of an explorer cell the CPOM cells must cover
RATE_STEP_M_PER_YR = 0.001
CHANGE_STEP_M = 0.01
REGIONS = ("AIS", "WAIS", "EAIS", "APIS")
REGION_NAMES = {
    "AIS": "Antarctic Ice Sheet",
    "WAIS": "West Antarctic Ice Sheet",
    "EAIS": "East Antarctic Ice Sheet",
    "APIS": "Antarctic Peninsula Ice Sheet",
}
# Orbit limits of the missions: ERS-2 and Envisat reach 81.5 S, CryoSat-2 88 S. CPOM fills
# the gap south of each by extrapolating the surrounding trends (Shepherd et al. 2019).
POLE_HOLE_LATITUDES = {"ers_envisat": -81.5, "cryosat2": -88.0}
TERRAIN_PACKAGES = {"480": "bedmachine_antarctica_v4_480", "741": "bedmachine_antarctica_v4_741"}
RATE_BASENAME = "cpom_antarctica_dhdt"
CHANGE_BASENAME = "cpom_antarctica_dh_480"
EPOCH_FILE = re.compile(r"CPOM-AIS-L3C-DH-MULTIMISSION-5KM-(\d{8})-(\d{8})-fv1\.nc$")

SOURCES = {
    "dataset": {
        "citation": "Centre for Polar Observation and Modelling (2026): Antarctic Ice Sheet surface elevation change at 5 km from multi-mission radar altimetry (ERS-2, Envisat, CryoSat-2), product version 1.0. CPOM Land Ice Portal.",
        "url": "https://www.cpom.ucl.ac.uk/landice/",
        "license": "CC-BY-4.0",
        "creators": "A. Muir, T. Slater, J. Maddalena, B. Palmer (CPOM)",
    },
    "paper": {
        "citation": "Shepherd, A. et al. (2019): Trends in Antarctic Ice Sheet elevation and mass. Geophysical Research Letters 46, 8174-8183.",
        "doi": "10.1029/2019GL082182",
    },
    "missions": "ERS-2 and Envisat (ESA FDR4ALT Land Ice Thematic Data Product v1) and CryoSat-2 (ESA Cryo-TEMPO Land Ice, Baseline D)",
}


# ---------------------------------------------------------------------------- inputs


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_terrain(meta_path: Path) -> tuple[dict, np.ndarray]:
    """Grid and BedMachine mask (uint8, rows north to south) of an explorer terrain package."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    raw = Path(meta_path).with_name(Path(meta_path).name.replace(".meta.json", ".bin")).read_bytes()
    grid = {key: meta["grid"][key] for key in ("nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m")}
    field = next(f for f in meta["fields"] if f["name"] == "mask")
    mask = np.frombuffer(raw, dtype=np.uint8, count=field["byte_length"], offset=field["byte_offset"])
    return grid, mask.reshape(grid["ny"], grid["nx"]).copy()


def read_cpom_field(path: Path, names: tuple[str, ...]) -> dict:
    """Axes, the named 2-D fields (NaN outside the data) and attributes, rows north to south."""
    import netCDF4  # noqa: PLC0415 - only the packing run needs it

    with netCDF4.Dataset(path) as ds:
        x = np.ma.filled(ds["x"][:], np.nan).astype(np.float64)
        y = np.ma.filled(ds["y"][:], np.nan).astype(np.float64)
        fields = {name: np.ma.filled(ds[name][:].astype(np.float64), np.nan).squeeze() for name in names}
        times = {
            key: float(np.ma.filled(ds[key][:], np.nan).squeeze())
            for key in ("start_time", "end_time")
            if key in ds.variables
        }
        attrs = {key: str(ds.getncattr(key)) for key in ("tracking_id", "date_created", "product_version", "license") if key in ds.ncattrs()}
        basin = np.ma.filled(ds["basin_id"][:], 0).astype(np.int16) if "basin_id" in ds.variables else None
    if not (np.all(np.diff(x) > 0) and np.all(np.abs(np.diff(x) - CPOM_CELL_M) < 1e-3)):
        raise ValueError(f"{path.name}: x is not an increasing 5 km axis")
    if not np.all(np.abs(np.abs(np.diff(y)) - CPOM_CELL_M) < 1e-3):
        raise ValueError(f"{path.name}: y is not a 5 km axis")
    if y[0] < y[-1]:  # CPOM rows run south to north; the explorer's run north to south
        y = y[::-1]
        fields = {name: values[::-1] for name, values in fields.items()}
        basin = basin[::-1] if basin is not None else None
    return {"x": x, "y": y, "fields": fields, "times": times, "attrs": attrs, "basin_id": basin}


def list_epochs(directory: Path) -> list[dict]:
    """The cumulative files in time order, checked to share the 21 March 1995 reference."""
    epochs = []
    for path in sorted(Path(directory).glob("CPOM-AIS-L3C-DH-MULTIMISSION-5KM-*-fv1.nc")):
        match = EPOCH_FILE.search(path.name)
        if not match:
            continue
        start, end = match.groups()
        if start != "19950321":
            raise ValueError(f"{path.name}: the record should start on 21 March 1995")
        epochs.append({"path": path, "end_date": end})
    if len(epochs) < 2:
        raise ValueError(f"fewer than two cumulative files in {directory}")
    return sorted(epochs, key=lambda epoch: epoch["end_date"])


def read_series(directory: Path) -> dict:
    """CPOM's cumulative mass change (Gt) and its 1-sigma uncertainty, per region and epoch."""
    series = {}
    for region in REGIONS:
        rows = json.loads((Path(directory) / f"dmdt_{region}.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"dmdt_{region}.json holds no series")
        series[region] = {
            "years": np.array([float(row["time"]) for row in rows]),
            "mass_change_gt": np.array([float(row["dm"]) for row in rows]),
            "uncertainty_gt": np.array([float(row["uncertainty"]) for row in rows]),
        }
    return series


# ---------------------------------------------------------------------------- resampling


def overlap_weights(src_centers: np.ndarray, src_width: float, dst_centers: np.ndarray, dst_width: float) -> np.ndarray:
    """Fraction of each target cell's width covered by each source cell, shape (n_dst, n_src).

    The same separable overlap as `regrid_ismip6_2300_run.py`.
    """
    src = np.asarray(src_centers, dtype=np.float64)
    dst = np.asarray(dst_centers, dtype=np.float64)
    lo = np.maximum((dst - dst_width / 2)[:, None], (src - src_width / 2)[None, :])
    hi = np.minimum((dst + dst_width / 2)[:, None], (src + src_width / 2)[None, :])
    return np.clip(hi - lo, 0.0, None) / dst_width


def explorer_axes(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    x = grid["x0_m"] + grid["dx_m"] * np.arange(grid["nx"], dtype=np.float64)
    y = grid["y0_m"] + grid["dy_m"] * np.arange(grid["ny"], dtype=np.float64)
    return x, y


class AreaMean:
    """Area-weighted means of 5 km CPOM fields over the cells of one explorer grid."""

    def __init__(self, cpom_x: np.ndarray, cpom_y: np.ndarray, grid: dict):
        tx, ty = explorer_axes(grid)
        self.wx = overlap_weights(cpom_x, CPOM_CELL_M, tx, abs(grid["dx_m"]))
        self.wy = overlap_weights(cpom_y, CPOM_CELL_M, ty, abs(grid["dy_m"]))
        self.shape = (grid["ny"], grid["nx"])

    def cover(self, valid: np.ndarray) -> np.ndarray:
        """Share of each explorer cell covered by valid CPOM cells."""
        return self.wy @ valid.astype(np.float64) @ self.wx.T

    def mean(self, field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cover = self.cover(valid)
        total = self.wy @ np.where(valid, field, 0.0) @ self.wx.T
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cover > 0, total / cover, np.nan)
        return mean, cover


def nearest_source_index(cpom_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    """Index of the CPOM cell containing each explorer node along one axis (-1 outside)."""
    step = cpom_axis[1] - cpom_axis[0]
    index = np.rint((target_axis - cpom_axis[0]) / step).astype(np.int64)
    return np.where((index >= 0) & (index < cpom_axis.size), index, -1)


def sample_basins(basin_id: np.ndarray, cpom_x: np.ndarray, cpom_y: np.ndarray, grid: dict) -> np.ndarray:
    """IMBIE basin of the CPOM cell under each explorer node (0 where there is none)."""
    tx, ty = explorer_axes(grid)
    col = nearest_source_index(cpom_x, tx)
    row = nearest_source_index(cpom_y, ty)
    out = np.zeros((ty.size, tx.size), dtype=np.uint8)
    rows, cols = np.ix_(row, col)
    inside = (rows >= 0) & (cols >= 0)
    values = basin_id[np.clip(rows, 0, None), np.clip(cols, 0, None)]
    out[inside] = np.clip(values[inside], 0, 255).astype(np.uint8)
    return out


def drawn_cells(cover: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Explorer ice cells that CPOM's cells cover by at least MIN_COVER."""
    return (cover >= MIN_COVER) & np.isin(mask, EXPLORER_ICE_FLAGS)


def pole_hole_radius_m(latitude_deg: float) -> float:
    """Distance from the pole of a latitude circle in EPSG:3031 (WGS84, true scale at 71 S)."""
    a, e = 6378137.0, 0.0818191908426215

    def t(phi: float) -> float:
        return np.tan(np.pi / 4 - phi / 2) / ((1 - e * np.sin(phi)) / (1 + e * np.sin(phi))) ** (e / 2)

    phi, phic = np.deg2rad(-latitude_deg), np.deg2rad(71.0)
    mc = np.cos(phic) / np.sqrt(1 - (e * np.sin(phic)) ** 2)
    return float(a * mc * t(phi) / t(phic))


# ---------------------------------------------------------------------------- packaging


def quantize(values: np.ndarray, step: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.full(array.shape, INT16_FILL, dtype=np.int16)
    valid = np.isfinite(array)
    codes = np.rint(array[valid] / step)
    if codes.size and np.max(np.abs(codes)) > INT16_MAX:
        raise ValueError(f"values up to {np.max(np.abs(array[valid])):.3f} do not fit int16 at a step of {step}")
    out[valid] = codes.astype(np.int16)
    return out


def finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}


def pack_fields(specs: list[dict]) -> tuple[bytes, list[dict]]:
    """Concatenate field blobs in order and describe each (offsets tile the payload)."""
    blobs, fields, offset = [], [], 0
    for spec in specs:
        blob = spec.pop("blob")
        fields.append({**spec, "byte_offset": offset, "byte_length": len(blob)})
        blobs.append(blob)
        offset += len(blob)
    return b"".join(blobs), fields


def int16_field(name: str, values: np.ndarray, step: float, unit: str, stats_key: str) -> dict:
    return {
        "name": name,
        "dtype": "int16",
        "unit": unit,
        "scale": step,
        "offset": 0.0,
        "fill_value": INT16_FILL,
        f"stats_{stats_key}": finite_stats(values),
        "blob": quantize(values, step).astype("<i2").tobytes(),
    }


def write_package(output_dir: Path, basename: str, payload: bytes, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.bin").write_bytes(payload)
    (output_dir / f"{basename}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def common_meta(grid: dict, terrain_name: str) -> dict:
    return {
        "product_version": "CPOM multi-mission radar altimetry surface elevation change, Antarctica, 5 km, product version 1.0 (fv1)",
        "reference": SOURCES["paper"]["citation"] + " doi:" + SOURCES["paper"]["doi"],
        "license": "CC BY 4.0 (the CPOM files' licence); cite the sources listed under `sources`",
        "source_url": SOURCES["dataset"]["url"],
        "sources": SOURCES,
        "grid": dict(grid),
        "source_package": {"grid_from": f"{terrain_name}.meta.json", "mask_from": f"{terrain_name}.bin"},
        "resampling": {
            "method": "area-weighted mean of the 5 km CPOM cells overlapping each explorer cell",
            "min_cover": MIN_COVER,
            "drawn_on": "explorer ice cells (BedMachine Antarctica v4 grounded ice, floating ice and Lake Vostok)",
            "reason": "CPOM's cells are on a 5 km EPSG:3031 grid of their own; overlay and terrain cells must correspond one to one",
            "rows": "CPOM rows run south to north and are flipped",
        },
    }


def rate_package(rate: dict, rate_path: Path, grid: dict, mask: np.ndarray, terrain_name: str) -> tuple[bytes, dict]:
    sec, sigma = rate["fields"]["sec"], rate["fields"]["sec_uncertainty"]
    valid = np.isfinite(sec)
    remap = AreaMean(rate["x"], rate["y"], grid)
    mean, cover = remap.mean(sec, valid)
    sigma_mean, _ = remap.mean(np.where(np.isfinite(sigma), sigma, 0.0), valid)
    drawn = drawn_cells(cover, mask)
    dhdt = np.where(drawn, mean, np.nan)
    dhdt_sigma = np.where(drawn, sigma_mean, np.nan)
    basins = np.where(drawn, sample_basins(rate["basin_id"], rate["x"], rate["y"], grid), 0).astype(np.uint8)

    cell_area = abs(grid["dx_m"] * grid["dy_m"])
    source_rate = {"net": round(float(np.nansum(sec) * CPOM_CELL_M**2 / 1e9), 3), **gross_volume_km3(sec, CPOM_CELL_M**2)}
    packed_rate = {"net": round(float(np.nansum(dhdt) * cell_area / 1e9), 3), **gross_volume_km3(dhdt, cell_area)}
    payload, fields = pack_fields([
        int16_field("dhdt", dhdt, RATE_STEP_M_PER_YR, "m/yr", "m_per_yr"),
        int16_field("dhdt_uncertainty", dhdt_sigma, RATE_STEP_M_PER_YR, "m/yr", "m_per_yr"),
        {"name": "basin_id", "dtype": "uint8", "blob": basins.tobytes(),
         "flags": {"0": "none", "1-18": "IMBIE drainage basin of Rignot et al. (2016), as CPOM numbers them"}},
    ])
    meta = {
        "title": "CPOM Antarctic surface elevation change rate, 1995-2026, multi-mission radar altimetry",
        **common_meta(grid, terrain_name),
        "source_file": {"name": rate_path.name, "sha256": sha256_of(rate_path), **rate["attrs"]},
        "period": {"start_year": rate["times"].get("start_time"), "end_year": rate["times"].get("end_time"),
                   "note": "linear trend of the cross-calibrated, gap-filled elevation record over the period"},
        "pole_hole": {
            "radius_m": {key: round(pole_hole_radius_m(lat), 1) for key, lat in POLE_HOLE_LATITUDES.items()},
            "latitude_deg": POLE_HOLE_LATITUDES,
            "note": "south of 88 S the rate is extrapolated throughout; south of 81.5 S so are its years before CryoSat-2 data arrive, late in 2010",
        },
        "quantization": {"unit": "m/yr", "scale": RATE_STEP_M_PER_YR, "offset": 0.0, "int16_fill_value": INT16_FILL},
        "visualization": {"scale": "asinh", "linear_core_m_per_yr": 0.05, "saturation_m_per_yr": 5.0, "fade_below_sigma": 2.0},
        "coverage": {
            "drawn_cells": int(drawn.sum()),
            "by_mask": {str(flag): int((drawn & (mask == flag)).sum()) for flag in EXPLORER_ICE_FLAGS},
            "explorer_ice_cells_without_value": int((np.isin(mask, EXPLORER_ICE_FLAGS) & ~drawn & (mask != 3)).sum()),
        },
        "validation": {
            "note": "volume-change rate (grid area, no map-scale correction) of the CPOM 5 km field and of the packed cells: net, and over the thickening and the thinning cells. They differ where CPOM cells fall outside explorer ice or below the cover threshold, and averaging cancels opposite-signed rates within a cell",
            "source_km3_per_yr": source_rate,
            "packed_km3_per_yr": packed_rate,
        },
        "fields": fields,
    }
    return payload, meta


def epoch_frames(epochs: list[dict], grid: dict, mask: np.ndarray) -> dict:
    """Read every epoch, resample it, and fix the domain from the last one."""
    changes, starts, ends, files = [], [], [], []
    remap = None
    source_volume_km3 = []
    final_sigma = None
    for index, epoch in enumerate(epochs):
        data = read_cpom_field(epoch["path"], ("dH", "dH_uncertainty"))
        if remap is None:
            remap = AreaMean(data["x"], data["y"], grid)
        dh = data["fields"]["dH"]
        valid = np.isfinite(dh)
        mean, cover = remap.mean(dh, valid)
        changes.append(np.where(cover >= MIN_COVER, mean, np.nan))
        if index == len(epochs) - 1:
            sigma = data["fields"]["dH_uncertainty"]
            sigma_mean, _ = remap.mean(np.where(np.isfinite(sigma), sigma, 0.0), valid)
            final_sigma = np.where(cover >= MIN_COVER, sigma_mean, np.nan)
            source_gross = gross_volume_km3(dh, CPOM_CELL_M**2)
        starts.append(data["times"]["start_time"])
        ends.append(data["times"]["end_time"])
        source_volume_km3.append(float(np.nansum(dh) * CPOM_CELL_M**2 / 1e9))
        files.append({"name": epoch["path"].name, "sha256": sha256_of(epoch["path"]), **data["attrs"]})
    domain = np.isin(mask, EXPLORER_ICE_FLAGS) & np.isfinite(changes[-1])
    return {
        "domain": domain,
        "change": np.stack([frame[domain] for frame in changes]),
        "final_sigma": final_sigma[domain],
        "start_years": np.array(starts),
        "end_years": np.array(ends),
        "source_volume_km3": np.array(source_volume_km3),
        "source_gross_km3": source_gross,
        "files": files,
    }


def gross_volume_km3(change: np.ndarray, cell_area_m2: float) -> dict[str, float]:
    """Volumes of the thickening and of the thinning cells (km3)."""
    values = np.asarray(change, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "thickening": round(float(values[values > 0].sum() * cell_area_m2 / 1e9), 1),
        "thinning": round(float(values[values < 0].sum() * cell_area_m2 / 1e9), 1),
    }


def first_observed_epoch(frames: dict, grid: dict) -> int:
    """First epoch whose change since the previous one varies across 82-87 S.

    Before CryoSat-2, CPOM extrapolates region-average trends south of 81.5 S, so the
    increments there are uniform within each extrapolation region; CryoSat-2 data bring
    spatial structure. Returns the index of the first epoch with that structure.
    """
    tx, ty = explorer_axes(grid)
    radius = np.hypot(*np.meshgrid(tx, ty))[frames["domain"]]
    annulus = (radius > pole_hole_radius_m(-87.0)) & (radius < pole_hole_radius_m(-82.0))
    increments = np.diff(frames["change"][:, annulus], axis=0)
    spread = np.nanpercentile(increments, 95, axis=1) - np.nanpercentile(increments, 5, axis=1)
    reference = np.nanmedian(spread[-20:])
    structured = np.flatnonzero(spread > 0.25 * reference)
    return int(structured[0] + 1) if structured.size else 0


def change_package(frames: dict, series: dict, grid: dict, terrain_name: str) -> tuple[bytes, dict]:
    centres = (frames["start_years"] + frames["end_years"]) / 2.0
    if np.any(np.diff(centres) <= 0):
        raise ValueError("epoch centres must increase")
    for region, values in series.items():
        if values["years"].size != centres.size or np.max(np.abs(values["years"] - centres)) > 0.01:
            raise ValueError(f"the {region} series does not have the epochs of the cumulative files")
    domain = frames["domain"]
    count = int(domain.sum())
    cell_area = abs(grid["dx_m"] * grid["dy_m"])
    packed_volume = np.nansum(frames["change"], axis=1) * cell_area / 1e9
    packed_gross = gross_volume_km3(frames["change"][-1], cell_area)
    observed_from = first_observed_epoch(frames, grid)
    payload, fields = pack_fields([
        int16_field("surface_change", frames["change"], CHANGE_STEP_M, "m", "m"),
        int16_field("final_uncertainty", frames["final_sigma"], CHANGE_STEP_M, "m", "m"),
        {"name": "domain_mask", "dtype": "uint8", "blob": domain.astype(np.uint8).tobytes(),
         "flags": {"0": "outside the domain", "1": "explorer ice cell with a CPOM value in the last epoch"}},
    ])
    fields[1]["note"] = (
        "1-sigma uncertainty of the cumulative change at the last epoch, one value per domain cell; "
        "CPOM's per-epoch uncertainty is zero for most cells before 2002, so it is not stored per frame"
    )
    meta = {
        "title": "CPOM Antarctic cumulative surface elevation change since March 1995, multi-mission radar altimetry",
        **common_meta(grid, terrain_name),
        "geometry_type": "sparse_grid_time_series",
        "time_series_kind": "observed_surface_change",
        "source_files": frames["files"],
        "domain": {"cell_count": count, "order": "row-major order of the non-zero cells of domain_mask"},
        "frames": {
            "count": int(centres.size),
            "years": [round(float(v), 4) for v in centres],
            "start_years": [round(float(v), 4) for v in frames["start_years"]],
            "end_years": [round(float(v), 4) for v in frames["end_years"]],
            "layout": "frame-major: surface_change[frame * cell_count + cell]; final_uncertainty and the domain have one value per cell",
            "note": "each frame is one 140-day epoch, placed at its centre, with CPOM's published change relative to its 1995 reference level; the first epoch departs from that level and is not subtracted",
        },
        "series": {
            "years": [round(float(v), 4) for v in centres],
            "gt_per_mm_sle": GT_PER_MM_SLE,
            "regions": {
                region: {
                    "name": REGION_NAMES[region],
                    "mass_change_gt": [round(float(v), 2) for v in series[region]["mass_change_gt"]],
                    "uncertainty_gt": [round(float(v), 2) for v in series[region]["uncertainty_gt"]],
                }
                for region in REGIONS
            },
            "note": "CPOM's cumulative mass change of the grounded ice sheet and of the IMBIE regions, 1-sigma, from the portal's basin time series; sea-level equivalent at 362.5 Gt per mm",
        },
        "pole_hole": {
            "radius_m": {key: round(pole_hole_radius_m(lat), 1) for key, lat in POLE_HOLE_LATITUDES.items()},
            "latitude_deg": POLE_HOLE_LATITUDES,
            "observed_from_year": round(float(frames["start_years"][observed_from]), 4),
            "note": "south of 81.5 S the change is extrapolated until CryoSat-2 data arrive (the first epoch whose increments vary across 82-87 S); south of 88 S it is extrapolated throughout",
        },
        "visualization": {
            "scale": "asinh", "linear_core_m": 1.0, "saturation_m": 150.0, "fade_below_sigma": 2.0,
            "fade_note": "a frame's change is faded where it is smaller than twice final_uncertainty",
        },
        "quantization": {"unit": "m", "scale": CHANGE_STEP_M, "offset": 0.0, "int16_fill_value": INT16_FILL},
        "validation": {
            "note": "volume change relative to CPOM's 1995 reference level at the last epoch (grid area, no map-scale correction), over every CPOM cell and over the packed domain cells. The net is a small residual of thickening and thinning that nearly cancel, so it moves with the margins; averaging to 10 km also cancels opposite-signed changes within a cell, which lowers both gross volumes",
            "end_source_km3": {"net": round(float(frames["source_volume_km3"][-1]), 1), **frames["source_gross_km3"]},
            "end_packed_km3": {"net": round(float(packed_volume[-1]), 1), **packed_gross},
        },
        "fields": fields,
    }
    return payload, meta


# ---------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cumulative-dir", type=Path, required=True, help="directory of the 81 cumulative dH files")
    parser.add_argument("--rate-file", type=Path, required=True, help="CPOM-AIS-L3C-DHDT-MULTIMISSION-5KM-...-fv1.nc")
    parser.add_argument("--series-dir", type=Path, required=True, help="directory holding dmdt_{AIS,WAIS,EAIS,APIS}.json")
    parser.add_argument("--terrain-dir", type=Path, default=Path("static/tools/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("static/tools/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rate = read_cpom_field(args.rate_file, ("sec", "sec_uncertainty"))
    for suffix, terrain_name in TERRAIN_PACKAGES.items():
        grid, mask = read_terrain(args.terrain_dir / f"{terrain_name}.meta.json")
        payload, meta = rate_package(rate, args.rate_file, grid, mask, terrain_name)
        write_package(args.output_dir, f"{RATE_BASENAME}_{suffix}", payload, meta)
        print(f"{RATE_BASENAME}_{suffix}: {meta['coverage']['drawn_cells']} cells, {len(payload) / 1e6:.1f} MB, "
              f"net {meta['validation']['packed_km3_per_yr']['net']} of {meta['validation']['source_km3_per_yr']['net']} km3/yr")

    grid, mask = read_terrain(args.terrain_dir / f"{TERRAIN_PACKAGES['480']}.meta.json")
    frames = epoch_frames(list_epochs(args.cumulative_dir), grid, mask)
    payload, meta = change_package(frames, read_series(args.series_dir), grid, TERRAIN_PACKAGES["480"])
    write_package(args.output_dir, CHANGE_BASENAME, payload, meta)
    print(f"{CHANGE_BASENAME}: {meta['frames']['count']} epochs x {meta['domain']['cell_count']} cells, "
          f"{len(payload) / 1e6:.1f} MB; observed south of 81.5 S from {meta['pole_hole']['observed_from_year']}")


if __name__ == "__main__":
    main()
