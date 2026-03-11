#!/usr/bin/env python3
"""Plot depth-averaged WAOM2 horizontal current speed around Antarctica."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from prepare_antarctica_ocean_currents import (
    build_cavity_margin_mask,
    build_main_antarctic_ice_mask,
    compute_depth_levels,
    gather_rho_to_u,
    gather_rho_to_v,
    interpolate_volume_to_target_depths,
    load_bedmachine_mask,
    read_masked_array,
)

DEFAULT_INPUT = "/Users/eddie/Documents/Antarctica_dataset/ocean_avg_yr5_annual.nc"
DEFAULT_BED_META = "static/tools/data/bedmachine_antarctica_v4_741.meta.json"
DEFAULT_BED_BIN = "static/tools/data/bedmachine_antarctica_v4_741.bin"
DEFAULT_OUTPUT_FULL = "output/figures/antarctica_depth_averaged_ocean_speed.png"
DEFAULT_OUTPUT_CAVITY80 = "output/figures/antarctica_depth_averaged_ocean_speed_cavity_margin80km.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="WAOM2 annual-mean NetCDF input.")
    parser.add_argument("--bedmachine-meta", default=DEFAULT_BED_META, help="BedMachine Antarctica metadata JSON.")
    parser.add_argument("--bedmachine-bin", default=DEFAULT_BED_BIN, help="BedMachine Antarctica binary.")
    parser.add_argument("--sample-stride", type=int, default=2, help="Subsample the WAOM grid for plotting.")
    parser.add_argument("--output-full", default=DEFAULT_OUTPUT_FULL, help="Output PNG for the full-ocean map.")
    parser.add_argument(
        "--output-cavity80",
        default=DEFAULT_OUTPUT_CAVITY80,
        help="Output PNG for the cavity+80 km masked map.",
    )
    return parser.parse_args()


def plot_speed(
    speed_mps: np.ndarray,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    finite = speed_mps[np.isfinite(speed_mps)]
    if finite.size == 0:
        raise RuntimeError(f"No finite speed values available for {output_path.name}.")
    vmax = float(np.percentile(finite, 99.0))
    vmax = max(vmax, float(np.max(finite)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.05

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e9edf2")

    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=200)
    image = ax.imshow(
        speed_mps,
        origin="lower",
        extent=[
            float(x_coords_m[0]) / 1000.0,
            float(x_coords_m[-1]) / 1000.0,
            float(y_coords_m[0]) / 1000.0,
            float(y_coords_m[-1]) / 1000.0,
        ],
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Polar stereographic x (km)")
    ax.set_ylabel("Polar stereographic y (km)")
    ax.set_facecolor("#f3f5f7")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Depth-averaged horizontal speed (m/s)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    sample_stride = max(1, int(args.sample_stride))
    model_x0_m = -3_000_000.0
    model_y0_m = -3_000_000.0
    model_dx_m = 2_000.0
    model_dy_m = 2_000.0

    bed_meta, bed_mask = load_bedmachine_mask(Path(args.bedmachine_meta), Path(args.bedmachine_bin))
    bed_grid = bed_meta["grid"]
    bed_main_ice_mask = build_main_antarctic_ice_mask(bed_mask, bed_grid)

    with netCDF4.Dataset(input_path) as ds:
        hc = float(ds.variables["hc"][:].item())
        s_rho = np.asarray(ds.variables["s_rho"][:], dtype=np.float32)
        c_rho = np.asarray(ds.variables["Cs_r"][:], dtype=np.float32)
        s_w = np.asarray(ds.variables["s_w"][:], dtype=np.float32)
        c_w = np.asarray(ds.variables["Cs_w"][:], dtype=np.float32)

        h_full = read_masked_array(ds.variables["h"], (slice(None), slice(None)))
        zice_full = read_masked_array(ds.variables["zice"], (slice(None), slice(None)))
        zeta_full = read_masked_array(ds.variables["zeta"], (0, slice(None), slice(None)))
        mask_rho_full = read_masked_array(ds.variables["mask_rho"], (slice(None), slice(None)))
        angle_full = read_masked_array(ds.variables["angle"], (slice(None), slice(None)))

        row_idx = np.arange(0, int(ds.dimensions["eta_rho"].size), sample_stride, dtype=np.int32)
        col_idx = np.arange(0, int(ds.dimensions["xi_rho"].size), sample_stride, dtype=np.int32)
        h = h_full[np.ix_(row_idx, col_idx)].copy()
        zice = zice_full[np.ix_(row_idx, col_idx)].copy()
        zeta = zeta_full[np.ix_(row_idx, col_idx)].copy()
        mask_rho = mask_rho_full[np.ix_(row_idx, col_idx)].copy()
        angle = angle_full[np.ix_(row_idx, col_idx)].copy()

        x_coords_m = model_x0_m + col_idx.astype(np.float64) * model_dx_m
        y_coords_m = model_y0_m + row_idx.astype(np.float64) * model_dy_m
        x_grid_m, y_grid_m = np.meshgrid(x_coords_m, y_coords_m)

        bed_col = np.rint((x_grid_m - float(bed_grid["x0_m"])) / float(bed_grid["dx_m"])).astype(np.int32)
        bed_row = np.rint((y_grid_m - float(bed_grid["y0_m"])) / float(bed_grid["dy_m"])).astype(np.int32)
        in_bed_extent = (
            (bed_col >= 0)
            & (bed_row >= 0)
            & (bed_col < int(bed_grid["nx"]))
            & (bed_row < int(bed_grid["ny"]))
        )
        bed_region_mask = np.full(mask_rho.shape, 255, dtype=np.uint8)
        bed_region_mask[in_bed_extent] = bed_mask[bed_row[in_bed_extent], bed_col[in_bed_extent]]
        bed_main_ice_region = np.zeros(mask_rho.shape, dtype=bool)
        bed_main_ice_region[in_bed_extent] = bed_main_ice_mask[bed_row[in_bed_extent], bed_col[in_bed_extent]]

        model_water_mask = (mask_rho > 0.5) & np.isfinite(h) & np.isfinite(zice)
        bed_water_mask = np.isin(bed_region_mask, (0, 3))
        open_water_mask = model_water_mask & bed_water_mask & (bed_region_mask == 0)
        cavity_water_mask = model_water_mask & bed_water_mask & (bed_region_mask == 3) & bed_main_ice_region
        water_mask = open_water_mask | cavity_water_mask
        cavity_margin80_mask = build_cavity_margin_mask(
            open_water_mask,
            cavity_water_mask,
            sample_dx_m=model_dx_m * sample_stride,
            sample_dy_m=model_dy_m * sample_stride,
            radius_m=80_000.0,
        )

        depth_levels = compute_depth_levels(h, zice, zeta, s_rho, c_rho, hc)[::-1].copy()
        w_depths = compute_depth_levels(h, zice, zeta, s_w, c_w, hc)[::-1].copy()
        layer_thickness = np.diff(w_depths, axis=0)

        level_count = s_rho.size
        ny_s, nx_s = h.shape
        xi_u_size = int(ds.dimensions["xi_u"].size)
        eta_v_size = int(ds.dimensions["eta_v"].size)
        u_left_native_cols = np.clip(col_idx - 1, 0, xi_u_size - 1).astype(np.int32)
        u_right_native_cols = np.clip(col_idx, 0, xi_u_size - 1).astype(np.int32)
        v_lower_native_rows = np.clip(row_idx - 1, 0, eta_v_size - 1).astype(np.int32)
        v_upper_native_rows = np.clip(row_idx, 0, eta_v_size - 1).astype(np.int32)
        needed_u_cols = np.unique(np.concatenate([u_left_native_cols, u_right_native_cols])).astype(np.int32)
        needed_v_rows = np.unique(np.concatenate([v_lower_native_rows, v_upper_native_rows])).astype(np.int32)
        u_col_lookup = {int(col): idx for idx, col in enumerate(needed_u_cols.tolist())}
        v_row_lookup = {int(row): idx for idx, row in enumerate(needed_v_rows.tolist())}
        u_left_lut = np.asarray([u_col_lookup[int(col)] for col in u_left_native_cols], dtype=np.int32)
        u_right_lut = np.asarray([u_col_lookup[int(col)] for col in u_right_native_cols], dtype=np.int32)
        v_lower_lut = np.asarray([v_row_lookup[int(row)] for row in v_lower_native_rows], dtype=np.int32)
        v_upper_lut = np.asarray([v_row_lookup[int(row)] for row in v_upper_native_rows], dtype=np.int32)

        depth_levels_u = compute_depth_levels(
            gather_rho_to_u(h_full, row_idx, needed_u_cols),
            gather_rho_to_u(zice_full, row_idx, needed_u_cols),
            gather_rho_to_u(zeta_full, row_idx, needed_u_cols),
            s_rho,
            c_rho,
            hc,
        )[::-1].copy()
        depth_levels_v = compute_depth_levels(
            gather_rho_to_v(h_full, needed_v_rows, col_idx),
            gather_rho_to_v(zice_full, needed_v_rows, col_idx),
            gather_rho_to_v(zeta_full, needed_v_rows, col_idx),
            s_rho,
            c_rho,
            hc,
        )[::-1].copy()

        u_native = np.full((level_count, ny_s, needed_u_cols.size), np.nan, dtype=np.float32)
        v_native = np.full((level_count, needed_v_rows.size, nx_s), np.nan, dtype=np.float32)
        for level in range(level_count):
            u_slice = read_masked_array(ds.variables["u"], (0, level, slice(None), slice(None)))
            v_slice = read_masked_array(ds.variables["v"], (0, level, slice(None), slice(None)))
            u_native[level] = u_slice[np.ix_(row_idx, needed_u_cols)]
            v_native[level] = v_slice[np.ix_(needed_v_rows, col_idx)]

    u_native = u_native[::-1].copy()
    v_native = v_native[::-1].copy()
    u_left = interpolate_volume_to_target_depths(u_native[:, :, u_left_lut], depth_levels_u[:, :, u_left_lut], depth_levels)
    u_right = interpolate_volume_to_target_depths(
        u_native[:, :, u_right_lut], depth_levels_u[:, :, u_right_lut], depth_levels
    )
    v_lower = interpolate_volume_to_target_depths(
        v_native[:, v_lower_lut, :], depth_levels_v[:, v_lower_lut, :], depth_levels
    )
    v_upper = interpolate_volume_to_target_depths(
        v_native[:, v_upper_lut, :], depth_levels_v[:, v_upper_lut, :], depth_levels
    )
    u_rho = 0.5 * (u_left + u_right)
    v_rho = 0.5 * (v_lower + v_upper)

    cos_angle = np.cos(angle).astype(np.float32)
    sin_angle = np.sin(angle).astype(np.float32)
    u_rot = u_rho * cos_angle[None, ...] - v_rho * sin_angle[None, ...]
    v_rot = u_rho * sin_angle[None, ...] + v_rho * cos_angle[None, ...]

    valid_layer = np.isfinite(layer_thickness) & (layer_thickness > 0)
    layer_thickness = np.where(valid_layer, layer_thickness, 0.0)
    weighted_u = np.where(np.isfinite(u_rot), u_rot * layer_thickness, 0.0)
    weighted_v = np.where(np.isfinite(v_rot), v_rot * layer_thickness, 0.0)
    thickness_sum = np.sum(layer_thickness, axis=0)
    u_bar = np.divide(weighted_u.sum(axis=0), thickness_sum, out=np.full_like(thickness_sum, np.nan), where=thickness_sum > 0)
    v_bar = np.divide(weighted_v.sum(axis=0), thickness_sum, out=np.full_like(thickness_sum, np.nan), where=thickness_sum > 0)
    speed_bar = np.hypot(u_bar, v_bar).astype(np.float32)
    speed_bar[~water_mask] = np.nan

    full_output = Path(args.output_full)
    cavity_output = Path(args.output_cavity80)
    plot_speed(
        speed_bar,
        x_coords_m,
        y_coords_m,
        title=f"WAOM2 annual-mean depth-averaged horizontal current speed (stride={sample_stride})",
        output_path=full_output,
    )
    cavity_speed = speed_bar.copy()
    cavity_speed[~cavity_margin80_mask] = np.nan
    plot_speed(
        cavity_speed,
        x_coords_m,
        y_coords_m,
        title=f"WAOM2 depth-averaged horizontal current speed in ice-shelf cavities + 80 km ocean (stride={sample_stride})",
        output_path=cavity_output,
    )

    print(full_output)
    print(cavity_output)


if __name__ == "__main__":
    main()
