"""Unit tests for VRTRestorer."""
from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from restorax.core.exceptions import RestorerLoadError
from restorax.restorers.super_resolution.vrt import VRTRestorer
from restorax.core.restorer import RestorerCategory

torch = pytest.importorskip("torch")


class TestVRTRestorerMeta:
    def test_name(self):
        r = VRTRestorer()
        assert r.name == "vrt_x4"

    def test_capabilities_category(self):
        caps = VRTRestorer().capabilities
        assert caps.category == RestorerCategory.SUPER_RESOLUTION

    def test_capabilities_scale_factor(self):
        caps = VRTRestorer().capabilities
        assert caps.scale_factor == 4

    def test_capabilities_requires_temporal(self):
        caps = VRTRestorer().capabilities
        assert caps.requires_temporal is True


class TestVRTBuildModelRaisesWhenBasicsrAbsent:
    def test_raises_restorer_load_error(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "basicsr" or name.startswith("basicsr."):
                raise ImportError("No module named 'basicsr'")
            return real_import(name, *args, **kwargs)

        device = torch.device("cpu")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError, match="VRT arch unavailable"):
                VRTRestorer._build_model(device)
