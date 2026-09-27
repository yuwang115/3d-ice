"""The preparation scripts' default arguments must rebuild the packages they are named after.

A script run with no options writes to its default basename, so if the defaults drifted
from the settings that produced the committed package, a clean rerun would silently
overwrite it with a package on a different grid from every overlay aligned to it.
"""

from __future__ import annotations

import json
import sys

import pytest


def _defaults(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__])
    return module.parse_args()


@pytest.mark.integration
def test_bedmachine_antarctica_defaults_match_the_committed_balanced_package(
    bedmachine_antarctica_module, data_dir, monkeypatch
):
    args = _defaults(bedmachine_antarctica_module, monkeypatch)
    meta = json.loads((data_dir / f"{args.basename}.meta.json").read_text(encoding="utf-8"))

    assert args.step == meta["downsample_step"]


@pytest.mark.integration
def test_bedmachine_greenland_defaults_match_the_committed_balanced_package(
    bedmachine_greenland_module, data_dir, monkeypatch
):
    args = _defaults(bedmachine_greenland_module, monkeypatch)
    meta = json.loads((data_dir / f"{args.basename}.meta.json").read_text(encoding="utf-8"))

    assert args.resolution_m == meta["target_resolution_m"]


@pytest.mark.integration
def test_bedmap3_defaults_match_the_committed_balanced_package(bedmap3_antarctica_module, data_dir, monkeypatch):
    args = _defaults(bedmap3_antarctica_module, monkeypatch)
    meta = json.loads((data_dir / f"{args.basename}.meta.json").read_text(encoding="utf-8"))

    assert args.resolution_m == meta["downsample_resolution_m"]
