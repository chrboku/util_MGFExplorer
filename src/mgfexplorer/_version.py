"""Package version, resolved without importing any other mgfexplorer modules."""


def _get_version() -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("util-mgfexplorer")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass

    # Fallback: walk up from this file and parse pyproject.toml
    import tomllib
    import pathlib

    here = pathlib.Path(__file__).parent
    for parent in (here, here.parent, here.parent.parent):
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            with open(candidate, "rb") as fh:
                data = tomllib.load(fh)
            return data.get("project", {}).get("version", "unknown")
    return "unknown"


__version__ = _get_version()
