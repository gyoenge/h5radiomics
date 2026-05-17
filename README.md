# HEST-Radiomics (hestradiomics)

Radiomics feature extraction pipeline for HEST dataset with WSI (Whole Slide Image) patches.

---

## Installation

Firstly, create and activate conda environment: 
```bash
conda create -n hestradiomics python=3.10  # cellvit requires >=3.10
conda activate hestradiomics 
```

Install `torch` and `torchvision`, matching with your cuda environment: 
```bash
# example 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0,>=1.24"  # for numpy degrading 
```

Then: 
```bash
# inside the hest-radiomics/ directory 
pip install -e . --no-build-isolation
```

We need `cellvit` module installation for `segment` engine: 
```bash
conda install -c conda-forge openslide
pip install openslide-python openslide-bin
pip install cellvit
```

---

## Run
```bash 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python src/hestradiomics/run.py
```

... 

