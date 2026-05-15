from dataclasses import dataclass, field 
from pathlib import Path 


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


# ============================================================
# HEST download setting 
# ============================================================

@dataclass(frozen=True)
class DownloadConfig: 
    root: Path = Path("./data").resolve()
    subroot: str = "hest"

    oncotrees: tuple[str] = field(default_factory=lambda: [
        "IDC", 
        "SKCM",
        "LUAD", 
        "PAAD", 
        "COAD", 
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
    num_workers: int = 0 

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
    save_patches: bool = False 
    num_workers: int = 0 

    output_dirname: str = "radiomics_features"
    segment_dirname: str = "cellseg"
    patch_dirname: str = "patches"

    overwrite: bool = False 


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


CONFIG = PipelineConfig()


# ============================================================

