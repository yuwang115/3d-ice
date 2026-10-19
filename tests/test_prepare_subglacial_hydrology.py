"""Channel edges of the subglacial hydrology packer.

GlaDS channel discharge is signed relative to the orientation of each mesh edge, so a
channel carrying 5 m3/s can be stored as -5. Version 0.3.0 kept only edges with
`discharge >= min_discharge`, dropping 35-42 % of the channel network by edge orientation
alone; the network rendered as broken dashes.
"""

from __future__ import annotations

import numpy as np


def _reduce(module, discharge, *, x1, y1, x2, y2, min_discharge=1e-3):
    return module.reduce_channel_edges(
        nx=4, ny=4, x0=0.0, y0=0.0, dx=1.0, dy=1.0,
        x1=np.asarray(x1, dtype=np.float64), y1=np.asarray(y1, dtype=np.float64),
        x2=np.asarray(x2, dtype=np.float64), y2=np.asarray(y2, dtype=np.float64),
        discharge=np.asarray(discharge, dtype=np.float64), min_discharge=min_discharge,
    )


def test_negative_discharge_is_a_channel_too(subglacial_hydrology_module):
    *_, q, kept = _reduce(subglacial_hydrology_module, [5.0, -7.0], x1=[0, 1], y1=[0, 1], x2=[1, 2], y2=[0, 1])
    assert kept == 2
    np.testing.assert_allclose(sorted(q.tolist()), [5.0, 7.0])


def test_threshold_applies_to_the_magnitude(subglacial_hydrology_module):
    *_, q, kept = _reduce(
        subglacial_hydrology_module, [-5e-4, 5e-4, -2e-3], x1=[0, 1, 2], y1=[0, 0, 0], x2=[1, 2, 3], y2=[0, 0, 0]
    )
    assert kept == 1
    np.testing.assert_allclose(q, [2e-3])


def test_duplicate_edges_keep_the_largest_magnitude(subglacial_hydrology_module):
    # Two mesh edges fall on the same pair of cells, once in each direction.
    *_, q, kept = _reduce(subglacial_hydrology_module, [3.0, -9.0], x1=[0, 1], y1=[0, 0], x2=[1, 0], y2=[0, 0])
    assert kept == 2
    np.testing.assert_allclose(q, [9.0])


def test_default_threshold_matches_the_shipped_packages(subglacial_hydrology_module):
    import sys

    argv = sys.argv
    sys.argv = ["prepare_subglacial_hydrology.py", "--input", "x.nc"]
    try:
        args = subglacial_hydrology_module.parse_args()
    finally:
        sys.argv = argv
    assert args.min_channel_discharge == 1e-3
