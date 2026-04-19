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


- run_extract.py에 mask_source/celltype_mode 옵션 추가
- run_full.py 순서를 segment -> extract -> statistics로 변경
- 기본값은 하위 호환 유지
    - mask_source="threshold"면 기존 방식 그대로
    - mask_source="cellseg"면 새 방식

run_extract.py
- 새 CLI 옵션 추가:
    - --mask_source
    - --cellseg_path
    - --celltype_mode
    - --target_cell_type
- sample별 cellseg.parquet 자동 경로 지원
- extract_radiomics(...) 호출 시 새 인자 전달

run_full.py
- extract 관련 새 옵션 추가
- config_to_cli_args_for_extract()에 새 옵션 전달
- 전체 실행 순서를 segment -> extract -> statistics로 변경


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

---

### 실행 예시

전체 파이프라인:

```bash
python -m h5radiomics.pipelines.run_full \
  --config configs/full.yaml
```

extract만:

```bash
python -m h5radiomics.pipelines.run_extract \
  --config configs/extract.yaml
```

특정 cell type만:

```bash
python -m h5radiomics.pipelines.run_extract \
  --config configs/extract.yaml \
  --mask_source cellseg \
  --celltype_mode single \
  --target_cell_type neoplastic
```
