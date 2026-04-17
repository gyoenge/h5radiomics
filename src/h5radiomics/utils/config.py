import copy
import yaml 

# Priority: DEFAULT < YAML < CLI 

# default configuration 
DEFAULT_CONFIG = {
    "sample_ids": ["TENX95", "NCBI785", "NCBI783", "TENX99"],
    "h5_dir": "/root/workspace/hest-radiomics/data/h5",
    "output_root": "/root/workspace/hest-radiomics/data/outputs",
    "label": 255,
    "save_patches": False,
    "classes": ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
    "filters": ["Original"],
    "num_workers": 0,
    "image_type_settings": {
        "LoG": {
            "sigma": [1.0, 2.0, 3.0]
        }
    },
    "processing": {
        "lower_q": 0.01,
        "upper_q": 0.99,
        "save_processed": True,
    },
}

def get_default_config():
    return DEFAULT_CONFIG


# yaml configuration 
def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return data


# merge whole configuration overriding cli arguments 
def merge_config(defaults, yaml_config, cli_args):
    # default merge (deep merge for dict)
    config = copy.deepcopy(defaults)

    # YAML merge (deep merge for dict)
    if yaml_config:
        for k, v in yaml_config.items():
            if v is None:
                continue

            if isinstance(v, dict) and isinstance(config.get(k), dict):
                # nested dict merge
                config[k].update(v)
            else:
                config[k] = v

    # CLI override
    for k, v in vars(cli_args).items():
        if k in ("config", "save_patches", "no_save_patches"):
            continue
        if v is not None:
            config[k] = v

    return config

