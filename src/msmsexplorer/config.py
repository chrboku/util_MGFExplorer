"""User configuration management for MGF Explorer."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List

CONFIG_DIR_NAME = ".msmsexplorer"
CONFIG_FILE_NAME = "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_grouping_tags": [],
    "metadata_groups": [],
}


def _get_config_dir() -> str:
    """Return the directory used to store the user configuration."""
    home_dir = os.path.expanduser("~")
    if not home_dir:
        home_dir = os.getcwd()
    return os.path.join(home_dir, CONFIG_DIR_NAME)


def get_config_path() -> str:
    """Return the full path to the configuration file."""
    return os.path.join(_get_config_dir(), CONFIG_FILE_NAME)


def _normalise_grouping_tags(raw_tags: Any) -> List[str]:
    """Normalise the grouping tags value into a cleaned list of strings."""
    if raw_tags is None:
        return []

    if isinstance(raw_tags, str):
        candidate = [part.strip() for part in raw_tags.split(",")]
    elif isinstance(raw_tags, list):
        candidate = [str(part).strip() for part in raw_tags]
    else:
        candidate = []

    return [tag for tag in candidate if tag]


def _normalise_metadata_groups(raw_groups: Any) -> List[Dict[str, Any]]:
    """Normalise the metadata groups structure into a predictable shape."""
    if not isinstance(raw_groups, list):
        return []

    normalised: List[Dict[str, Any]] = []
    for group in raw_groups:
        if not isinstance(group, dict):
            continue

        name = str(group.get("name", "")).strip()
        keys_raw = group.get("keys", [])

        if isinstance(keys_raw, str):
            keys = [part.strip() for part in keys_raw.split(",") if part.strip()]
        elif isinstance(keys_raw, list):
            keys = [str(part).strip() for part in keys_raw if str(part).strip()]
        else:
            keys = []

        normalised.append({"name": name or "", "keys": keys})

    return normalised


def _normalise_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required structure and value types for the configuration."""
    normalised = copy.deepcopy(DEFAULT_CONFIG)
    normalised.update({k: v for k, v in config.items() if k in DEFAULT_CONFIG})

    normalised["default_grouping_tags"] = _normalise_grouping_tags(
        normalised.get("default_grouping_tags")
    )
    normalised["metadata_groups"] = _normalise_metadata_groups(
        normalised.get("metadata_groups")
    )

    return normalised


def load_config() -> Dict[str, Any]:
    """Load the user configuration from disk, falling back to defaults."""
    config_path = get_config_path()
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return _normalise_config(data)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Warning: Failed to load config file {config_path}: {exc}")

    return copy.deepcopy(DEFAULT_CONFIG)


def save_config(config: Dict[str, Any]) -> None:
    """Persist the configuration to disk."""
    config_dir = _get_config_dir()
    os.makedirs(config_dir, exist_ok=True)

    config_path = get_config_path()
    normalised = _normalise_config(config)

    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(normalised, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
