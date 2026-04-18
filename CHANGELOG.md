## v0.2.0 

developing 

### In this commit 

기존:

- patch image
- grayscale threshold로 tissue mask 생성
- radiomics 1 row / patch

수정 후:

- patch image
- cellseg.parquet에서 해당 patch_idx의 polygon 로드
- class_name 기준으로 mask 생성
- radiomics 추출
    - celltype_mode="merged": patch 내 모든 cell 합쳐서 1 row
    - celltype_mode="per_class": patch × class 별 1 row
    - celltype_mode="single": 특정 class만 추출

즉, output row가 patch 하나당 여러 개가 될 수 있습니다.


### 이 수정으로 생기는 output 예시

celltype_mode="per_class"이면 row가 이런 식으로 나옵니다.

```python
{
  "patch_idx": 17,
  "barcode": "AACT...",
  "region_type": "neoplastic",
  "cell_type": "neoplastic",
  "n_cells": 24,
  "mask_area": 1832,
  "status": "ok",
  "original_firstorder_Mean": ...,
  ...
}
{
  "patch_idx": 17,
  "barcode": "AACT...",
  "region_type": "inflammatory",
  "cell_type": "inflammatory",
  "n_cells": 6,
  "mask_area": 421,
  "status": "ok",
  ...
}
```