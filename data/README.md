### Expected Data Directory Structure 

input_dir = "/root/workspace/hest-radiomics/data/h5"
output_dir = "/root/workspace/hest-radiomics/data/outputs"

outputs/
└── {sample_id}/
    ├── patches/
    │   ├── color/
    │   ├── gray/
    │   ├── mask/
    │   ├── masked_color/
    │   └── masked_gray/
    │
    ├── features/
    │   ├── raw/
    │   │   ├── features.csv
    │   │   └── features.parquet
    │   │
    │   ├── processed/
    │   │   ├── features.csv
    │   │   ├── features.parquet
    │   │   ├── processing_stats.csv
    │   │   └── processing_config.json   
    │   │
    │   └── statistics/
    │       ├── raw/
    │       │   ├── stats.csv
    │       │   ├── stats.parquet
    │       │   ├── representative/
    │       │   └── boxplots/
    │       │
    │       └── processed/
    │           ├── stats.csv
    │           ├── stats.parquet
    │           ├── representative/
    │           └── boxplots/
    │
    └── cellvitseg/
        ├── cellseg.geojson
        ├── cellseg.parquet
        ├── metadata.csv
        ├── summary.json
        └── overlay/

