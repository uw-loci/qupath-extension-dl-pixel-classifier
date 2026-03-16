"""Deep Learning Pixel Classifier Server for QuPath."""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("dlclassifier-server")
except Exception:
    __version__ = "0.3.5"  # fallback when running from JAR-bundled scripts
