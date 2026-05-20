from dataclasses import dataclass, field 
from pathlib import Path 
from typing import Optional, List


# ============================================================
# HEST dataset setting 
# ============================================================

@dataclass(frozen=True)
class HestConfig: 
    root: Path = Path("./data").resolve()
    subroot: str = "hestradiomics"

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

    # None: run all samples
    # list/tuple: run only selected samples
    sample_ids: Optional[tuple[str, ...]] = (
        # "NCBI783", 
        # "NCBI785", 
        "TENX95", 
        "TENX99"
    ) # None | ("NCBI681", "NCBI682")

    
# ============================================================
# Cell segmentation setting 
# ============================================================

@dataclass(frozen=True)
class SegmentConfig:
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
# Radiomics feature extraction setting 
# ============================================================

@dataclass(frozen=True)
class ExtractConfig: 
    mask_source: str = "cellseg" # "threshold" | "cellseg"

    output_dirname: str = "radiomics"
    segment_dirname: str = "segment"
    patch_dirname: str = "patches"

    num_workers: int = 16 # 0

    overwrite: bool = False 
    save_patches: bool = False 


# ============================================================
# Visualization setting
# ============================================================

@dataclass(frozen=True)
class VisualizeConfig: 
    vis_ratio = float = 0.02 # 0.0~1.0 
    overwrite: bool = True 


# ============================================================
# Statistics setting
# ============================================================

@dataclass(frozen=True)
class StatisticsConfig: 
    # output_dirname: str = "statistics"
    # overwrite: bool = True 
    save_boxplot: bool = True,
    save_representative: bool = True,
    representative_stats: List[str] = ["min", "q25", "q50", "q75", "max"],


# ============================================================
# Whole pipeline config 
# ============================================================

@dataclass(frozen=True)
class FullConfig:
    hest: HestConfig = field(default_factory=HestConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    @property
    def sample_ids(self) -> Optional[tuple[str, ...]]:
        return self.hest.sample_ids
    
    @property 
    def download_dir(self) -> Path:
        return self.hest.root / self.hest.subroot


CONFIG = FullConfig()


# ============================================================

