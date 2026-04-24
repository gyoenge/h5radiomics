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
  --config configs/full.yaml \
  --skip_segment --skip_statistics
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

---

### 모드에 따른 feature 추출 

- mask_source == "threshold"
  - 기존처럼 patch-level intensity/texture radiomics만 추출

- mask_source == "cellseg"
  - patch 1개당 row 1개
  - 아래를 모두 한 row에 넣음
    - patch-level intensity/texture
    - cellseg merged(all cells) intensity/texture
    - class별 cellseg intensity/texture
    - morphology(shape2D) per-cell 추출 후 first-order aggregation
    - cell-type count / ratio features


---

### extract engine 최적화

# 1. constants / dataclass
# 2. extractor builders
# 3. geometry helpers
# 4. naming / feature utility
# 5. patch loading / row builders
# 6. feature extractors
# 7. threshold / cellseg processors
# 8. chunk / pipeline runner
# 9. post-processing

1) process_single_patch 분리

지금은 patch 로딩, metadata, mask 생성, 저장, feature extraction, status handling이 다 섞여 있다. 

이걸 아래처럼 나눈다:

- load_patch_data()
- build_patch_row_base()
- process_threshold_patch()
- process_cellseg_patch()
- safe_update_features()

2) feature prefix 하드코딩 제거

지금 get_radiomics_feature_columns()에 class별 prefix가 전부 박혀 있음.

이건 이렇게 바꾼다: 

- radiomics feature suffix는 공통 규칙으로 판별
- prefix는 patch_, cellseg_, morph_ 같은 상위 prefix만 확인
- cellseg_neoplastic_, cellseg_dead_ 같은 건 동적으로 허용

3) extractor 생성 비용 줄이기

지금은 멀티프로세스에서 chunk마다 build_radiomics_extractor()를 새로 만듦.

이건 최소한 다음처럼 바꾸는 게 좋다:

- worker 단위 1회 생성
- chunk 함수 안이 아니라 worker initializer 또는 worker-local cache 사용

파이썬 ProcessPoolExecutor는 initializer 지원이 제한적일 수 있으니,
현 구조에서는 chunk 함수 내부 캐시 전역 변수로도 충분히 개선 가능.

4) I/O와 compute 분리

지금은 feature 계산 중에 이미지 저장도 같이 함.

이걸:

- compute 단계: feature, mask, metadata 생성
- save 단계: 필요한 경우만 이미지 저장

으로 나누면 테스트도 쉬워지고 재사용성도 좋아짐. 
