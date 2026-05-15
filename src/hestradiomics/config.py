### Run setting (All True for Full Pipeline)
RUN_HEST_DOWNLOAD = True 
RUN_CELL_SEGMENT = True  # Requires previous RUN_HEST_DOWNLOAD
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



### Extraction setting  



### (Optional) Statistic setting  
# if RUN_STATISTIC is True, we can use detailed settings following: 


