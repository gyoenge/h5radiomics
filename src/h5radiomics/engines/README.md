# extract_radiomics.py
# Radiomics feature extraction from HDF5 files containing image patches and associated metadata. 
"""
Example usage:

(i) Using YAML config file:
cd /root/workspace/h5radiomics/src 
python -m h5radiomics.extract_radiomics \
  --config ../configs/default.yaml \
  --num_workers 8

(ii) Using command-line arguments to override defaults or YAML config: 
python -m h5radiomics.extract_radiomics \
  --sample_ids TENX99 TENX95 NCBI785 NCBI783 \
  --h5_dir /root/workspace/h5radiomics/h5 \
  --output_root /root/workspace/h5radiomics/outputs \
  --label 255 \
  --save_patches \
  --classes firstorder glcm glrlm glszm gldm ngtdm \
  --filters Original \
  --num_workers 8 
""" 
# --filters Original Wavelet LoG Square SquareRoot Logarithm Exponential 


---


# feature_statistics.py
"""
Saved radiomics feature CSV files -> feature-wise statistics + representative patches

Example usage:

(i) Using YAML config file:
cd /root/workspace/h5radiomics/src
python -m h5radiomics.feature_statistics --config ../configs/stats.yaml

(ii) Using command-line arguments:
python -m h5radiomics.feature_statistics \
  --sample_ids TENX95 NCBI785 NCBI783 TENX99 \
  --input_root /root/workspace/h5radiomics/outputs \
  --output_root /root/workspace/h5radiomics/outputs/statistics \
  --status_filter ok \
  --save_representatives true \
  --save_boxplot true
"""


---


# segment_cellvit.py
"""
Example
-------
(i) Using YAML config file:
cd /root/workspace/h5radiomics/src
python -m h5radiomics.segment_cellvit \
  --config ../configs/segment_cellvit.yaml

(ii) Using command-line arguments:
python -m h5radiomics.segment_cellvit \
  --sample_ids TENX99 TENX95 NCBI783 NCBI785 \
  --input_dir /root/workspace/h5radiomics/h5 \
  --output_dir /root/workspace/h5radiomics/outputs/cellvit_patch_seg \
  --model_dir /root/workspace/h5radiomics/models \
  --model_name CellViT-SAM-H-x20.pth \
  --patch_indices 200 300 400 500 \
  --batch_size 8 \
  --num_workers 0 \
  --device cuda:0
"""
# --no_class_color
# --save_geojson_per_patch
# --postprocess_threads 1


