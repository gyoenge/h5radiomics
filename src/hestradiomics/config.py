### Run setting (All True for Full Pipeline)
RUN_HEST_DOWNLOAD = True 
RUN_SEGMENT = True
RUN_OVERLAY = True
RUN_RADIOMICS_EXTRACTION = True  # Requires previous RUN_HEST_DOWNLOAD & RUN_CELL_SEGMENT
RUN_STATISTICS = True  # Requires all previous (Optional)

### HEST dataset download setting 
DOWNLOAD_ROOT = "./data"
DOWNLOAD_SUBROOT = "hest"
DOWNLOAD_ONCOTREE = [
    "IDC", 
    "SKCM",
    "LUAD",
    "PAAD", 
    "COAD", 
]
DOWNLOAD_REQUIRED = [
    "patches",
    "st",
]
DOWNLOAD_OPTIONAL = [
    "metadata",
    "patches_vis",
    "thumbnails",
    "spatial_plots", 
] 
DOWNLOAD_TECH = [
    # "Spatial Transcriptomics" | "Visium HD" | "Visium" | "Xenium"
    "Xenium", 
]

### Cell-segment setting
MODEL_NAME = "CellViT-SAM-H-x20.pth"
MODEL_PATH = f"./models/{MODEL_NAME}"

DEVICE = "cuda:0"
BATCH_SIZE = 8
NUM_WORKERS = 0
USE_CLASS_COLOR = True

OVERWRITE_SEGMENT = False
OVERWRITE_OVERLAY = False

### Extraction setting  



### (Optional) Statistic setting  
# if RUN_STATISTIC is True, we can use detailed settings following: 


