# h5radiomics

> extracting radiomics features from HEST dataset using pyradiomics. 

### data inputs (need to prepare)
-  `h5/*.h5` : patches (HEST-1k style)

### expected outputs 
- `outputs/..` 

### run 
(i) Using YAML config file:
```bash
cd ./src 
python -m h5radiomics.extract_radiomics --config ../configs/default.yaml
```

(ii) Using command-line arguments to override defaults or YAML config: 
```bash 
cd ./src 
python -m h5radiomics.extract_radiomics \
  --sample_ids NCBI785 NCBI783 \
  --h5_dir /root/workspace/h5radiomics/h5 \
  --output_root /root/workspace/h5radiomics/output_test \
  --label 255 \
  --save_patches \
  --classes firstorder glcm glrlm glszm gldm ngtdm \
  --filters Original Wavelet LoG Square SquareRoot Logarithm Exponential
```
