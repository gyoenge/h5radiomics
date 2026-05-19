# Segment Output Format

    segment/
    ├── IDC/
    │   ├── TENX95.parquet
    │   ├── TENX99.parquet
    │   └── ...


각 parquet row:

| column           | description       |
| ---------------- | ----------------- |
| patch_idx        | patch index       |
| barcode          | ST barcode        |
| cell_id_in_patch | nucleus id        |
| class_id         | CellViT class     |
| class_name       | readable class    |
| geometry         | polygon           |
| coord_raw        | ST coord          |
| n_cells          | patch-level count |
