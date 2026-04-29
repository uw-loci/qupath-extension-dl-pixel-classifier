"""Deep Learning Pixel Classifier Server for QuPath."""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("dlclassifier-server")
except Exception:
    __version__ = "0.7.5-dev"  # fallback when running from JAR-bundled scripts
