from __future__ import annotations

import copy
import yaml


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return data


def deep_merge_dict(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)

    for k, v in override.items():
        if v is None:
            continue

        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)

    return result


def merge_config(defaults: dict, yaml_config: dict | None, cli_args, skip_keys: tuple[str, ...] = ("config",)) -> dict:
    config = copy.deepcopy(defaults)

    if yaml_config:
        config = deep_merge_dict(config, yaml_config)

    for k, v in vars(cli_args).items():
        if k in skip_keys:
            continue
        if v is not None:
            config[k] = v

    return config


def str_to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v

    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean value from: {v}")


def normalize_none_like(v):
    if v in ("None", "none", "", [], None):
        return None
    return v
