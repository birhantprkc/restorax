from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np

from restorax.dag.nodes.merge import MergeNode
from restorax.dag.nodes.parallel import BranchConfig


def test_merge_node_blend_averages_frames():
    frame_a = np.full((4, 4, 3), 100, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 200, dtype=np.uint8)
    # branch_outputs: 2 branches, 1 chunk each, 1 frame each
    branch_outputs = [[[frame_a]], [[frame_b]]]
    node = MergeNode(id="m1", name="Merge", strategy="blend")
    result = asyncio.run(node.execute(MagicMock(), {"branch_outputs": branch_outputs}))
    merged_frame = result.outputs["chunks"][0][0]
    assert merged_frame.dtype == np.uint8
    np.testing.assert_allclose(merged_frame, 150, atol=1)


def test_merge_node_select_picks_correct_branch():
    frame_a = np.full((4, 4, 3), 10, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 99, dtype=np.uint8)
    branch_outputs = [[[frame_a]], [[frame_b]]]
    node = MergeNode(id="m1", name="Merge", strategy="select", select_index=1)
    result = asyncio.run(node.execute(MagicMock(), {"branch_outputs": branch_outputs}))
    np.testing.assert_array_equal(result.outputs["chunks"][0][0], frame_b)


def test_branch_config_roundtrip():
    bc = BranchConfig(name="branch_a", restorer_steps=[("real_esrgan", {"scale": 4})])
    restored = BranchConfig.from_dict(bc.to_dict())
    assert restored.name == "branch_a"
    assert restored.restorer_steps[0][0] == "real_esrgan"
