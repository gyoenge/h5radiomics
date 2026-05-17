from dataclasses import dataclass, field 
from pathlib import Path 
from typing import Optional 


# ============================================================
# Run setting (All True for Full Pipeline)
# ============================================================

@dataclass(frozen=True)
class RunConfig:
    run_hest_download: bool = True 
    run_segment: bool = True 
    run_overlay: bool = True 
    run_radiomics_extraction: bool = True 
    run_statistics: bool = True 

    # None: run all samples
    # list/tuple: run only selected samples
    sample_ids: Optional[tuple[str, ...]] = None
    # sample_ids: Optional[tuple[str, ...]] = ("NCBI681", "NCBI682")


# ============================================================
# HEST download setting 
# ============================================================

@dataclass(frozen=True)
class DownloadConfig: 
    root: Path = Path("./data").resolve()
    subroot: str = "hest"

    oncotrees: tuple[str] = field(default_factory=lambda: [
        "IDC", 
        # "SKCM",
        # "LUAD", 
        # "PAAD", 
        # "COAD", 
    ])    

    required_dirs: tuple[str] = field(default_factory=lambda: [
        "patches", 
        "st", 
    ])

    optional_dirs: tuple[str] = field(default_factory=lambda: [
        "metadata",
        "patches_vis",
        "thumbnails",
        "spatial_plots",
    ])

    technologies: tuple[str] = field(default_factory=lambda: [
        # "Spatial Transcriptomics" | "Visium HD" | "Visium" | "Xenium"
        "Xenium", 
        "Visium", 
    ])

    @property 
    def download_dir(self) -> Path:
        return self.root / self.subroot

    
# ============================================================
# Cell segmentation setting 
# ============================================================

@dataclass(frozen=True)
class CellSegmentConfig:
    model_name: str = "CellViT-SAM-H-x20.pth"
    model_root: Path = Path("./models")

    device: str = "cuda:0"
    batch_size: int = 8 
    num_workers: int = 16 # 0

    use_class_color: bool = True 

    overwrite_segment: bool = False 
    overwrite_overlay: bool = False 

    @property 
    def model_path(self) -> Path:
        return self.model_root / self.model_name 
    

# ============================================================
# Radiomics extraction setting 
# ============================================================

@dataclass(frozen=True)
class RadiomicsConfig: 
    mask_source: str = "cellseg" # "threshold" | "cellseg"

    output_dirname: str = "radiomics"
    segment_dirname: str = "segment"
    patch_dirname: str = "patches"

    num_workers: int = 0 

    overwrite: bool = False 
    save_patches: bool = False 


# ============================================================
# Statistics setting
# ============================================================

@dataclass(frozen=True)
class StatisticsConfig: 
    output_dirname: str = "statistics"
    overwrite: bool = True 


# ============================================================
# Whole pipeline config 
# ============================================================

@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    cellseg: CellSegmentConfig = field(default_factory=CellSegmentConfig)
    radiomics: RadiomicsConfig = field(default_factory=RadiomicsConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    @property
    def sample_ids(self) -> Optional[tuple[str, ...]]:
        return self.run.sample_ids


CONFIG = PipelineConfig()


# ============================================================

