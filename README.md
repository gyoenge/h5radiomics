# h5radiomics

> extracting radiomics features from HEST dataset using pyradiomics. 

### data inputs (need to prepare)
-  `h5/*.h5` : patches (HEST-1k style)
-  `segmentation/*.parquet` : cell/nuclei segmentation masks (HEST-1k style)

### expected outputs 
- `output/..` 

### run 
(i) Using YAML config file:
```bash
cd src 
python extract_radiomics.py --config config.yaml
```

(ii) Using command-line arguments to override defaults or YAML config: 
```bash 
cd src 
python extract_radiomics.py \
  --sample_ids NCBI785 NCBI783 \
  --h5_dir /root/workspace/h5radiomics/h5 \
  --output_root /root/workspace/h5radiomics/output_test \
  --label 255 \
  --save_patches \
  --classes firstorder glcm glrlm glszm gldm ngtdm \
  --filters Original Wavelet LoG Square SquareRoot Logarithm Exponential
```
