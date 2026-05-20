from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.super_resolution.tdm import TDMRestorer


class TestTDMMeta:
    def test_name(self):
        assert TDMRestorer().name == "tdm"

    def test_category(self):
        assert TDMRestorer().capabilities.category == RestorerCategory.SUPER_RESOLUTION

    def test_requires_temporal(self):
        assert TDMRestorer().capabilities.requires_temporal is True

    def test_scale_factor(self):
        assert TDMRestorer().capabilities.scale_factor == 4


class TestTDMLoadRaisesWhenDiffusersAbsent:
    def test_raises_restorer_load_error_on_missing_diffusers(self, tmp_path):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "diffusers" or name.startswith("diffusers."):
                raise ImportError("No module named 'diffusers'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("restorax.config.settings.model_dir", str(tmp_path)),
        ):
            with pytest.raises(RestorerLoadError):
                TDMRestorer().load(torch.device("cpu"))
