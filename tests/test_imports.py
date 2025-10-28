def test_app_imports():
    """
    Basic smoke test to ensure app and key modules import
    """
    import importlib
    import app  # ensure top-level app imports successfully
    import agents
    import signatures
    assert app is not None
