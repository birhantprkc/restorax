import pytest


def test_requires_assets_mark_registered(pytestconfig):
    raw = pytestconfig.getini("markers")
    names = set()
    for m in raw:
        if hasattr(m, "name"):
            names.add(m.name)
        elif isinstance(m, str):
            names.add(m.split(":")[0].strip())
    assert "requires_assets" in names


@pytest.mark.requires_assets
def test_marked_test_can_be_collected(test_assets):
    # This will skip if assets aren't downloaded — that's the expected behavior
    assert test_assets.exists()
