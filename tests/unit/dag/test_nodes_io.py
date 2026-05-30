from __future__ import annotations

from restorax.dag.nodes.io import VideoInputNode, VideoOutputNode


def test_video_input_node_ports():
    node = VideoInputNode(id="vi", name="Input", video_path="/tmp/video.mp4")
    assert node.input_ports == []
    assert any(p.name == "chunks" for p in node.output_ports)
    assert any(p.name == "meta" for p in node.output_ports)


def test_video_input_node_roundtrip():
    node = VideoInputNode(id="vi", name="Input", video_path="/tmp/v.mp4", chunk_size=8)
    data = {"type": "video_input", "id": node.id, "name": node.name, **node.to_dict()}
    restored = VideoInputNode.from_dict(data)
    assert restored.chunk_size == 8
    assert restored.video_path == "/tmp/v.mp4"


def test_video_output_node_ports():
    node = VideoOutputNode(id="vo", name="Output")
    assert any(p.name == "chunks" for p in node.input_ports)
    assert any(p.name == "output_path" for p in node.output_ports)
