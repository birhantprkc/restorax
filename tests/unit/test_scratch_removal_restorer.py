from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from restorax.core.exceptions import RestorerLoadError
from restorax.core.restorer import RestorerCategory
from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer


class TestScratchRemovalMeta:
    def test_name(self):
        assert ScratchRemovalRestorer().name == "scratch_removal"

    def test_category(self):
        assert ScratchRemovalRestorer().capabilities.category == RestorerCategory.ARTIFACT_REMOVAL

    def test_requires_temporal(self):
        assert ScratchRemovalRestorer().capabilities.requires_temporal is True

    def test_scale_factor(self):
        assert ScratchRemovalRestorer().capabilities.scale_factor == 1


class TestScratchRemovalLoadRaisesWhenArchAbsent:
    def test_raises_restorer_load_error_on_missing_propainter(self):
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "propainter" in name.lower():
                raise ImportError("No module named 'propainter_arch'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RestorerLoadError):
                ScratchRemovalRestorer().load(torch.device("cpu"))


class TestProPainterPipelineVendored:
    """Smoke test for the vendored ProPainter arch (propainter_arch.py)."""

    def test_pipeline_constructs_with_matching_state_dicts(self, tmp_path):
        """
        ProPainterPipeline loads real checkpoints via strict state_dict loading.
        Rather than requiring a multi-GB download in CI, we save the state_dict
        of freshly-constructed (randomly initialized) real sub-networks and
        confirm ProPainterPipeline's real weight-loading path runs end-to-end.
        """
        import argparse

        from restorax.restorers.artifact_removal.propainter.propainter import InpaintGenerator
        from restorax.restorers.artifact_removal.propainter.raft.raft import RAFT
        from restorax.restorers.artifact_removal.propainter.recurrent_flow_completion import (
            RecurrentFlowCompleteNet,
        )
        from restorax.restorers.artifact_removal.propainter_arch import ProPainterPipeline

        raft_args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
        raft_state = {f"module.{k}": v for k, v in RAFT(raft_args).state_dict().items()}
        flow_state = RecurrentFlowCompleteNet().state_dict()
        inpaint_state = InpaintGenerator(init_weights=False).state_dict()

        for filename, state in [
            ("raft-things.pth", raft_state),
            ("recurrent_flow_completion.pth", flow_state),
            ("ProPainter.pth", inpaint_state),
        ]:
            torch.save(state, tmp_path / filename)

        pipeline = ProPainterPipeline(weight_dir=str(tmp_path), device=torch.device("cpu"))

        assert pipeline.model is not None
        assert pipeline.fix_flow_complete is not None
        assert pipeline.fix_raft is not None

    def test_pipeline_raises_when_weights_missing(self, tmp_path):
        from restorax.restorers.artifact_removal.propainter_arch import ProPainterPipeline

        with pytest.raises(FileNotFoundError):
            ProPainterPipeline(weight_dir=str(tmp_path), device=torch.device("cpu"))
