"""Test the working tree, not an installed copy.

The package is normally pip-installed non-editable, so a bare import would
resolve to site-packages and silently skip local edits.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class _StubClient:
    """Stands in for distributed.Client: _download only reads the dashboard port."""

    def __init__(self, *args, **kwargs):
        pass

    def scheduler_info(self):
        return {"services": {"dashboard": 0}}


@pytest.fixture(autouse=True)
def no_dask_cluster(monkeypatch):
    """Keep _download from spawning a real scheduler + worker processes."""
    import llc_cutout_dataloader.cutouts_dataset as cutouts_dataset
    monkeypatch.setattr(cutouts_dataset, "Client", _StubClient)
