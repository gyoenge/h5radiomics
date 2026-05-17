# from typing import Literal

# mode = Literal["light", "detailed"]

# def run(mode):
#     if mode == "light":
#         # run with no saving 
#         pass 
#     elif mode == "detailed":
#         # run with saving 
#         pass 


# if __name__ == "__main__":
#     run(mode="light")
    

from hestradiomics.config import CONFIG


def run():
    cfg = CONFIG

    # if cfg.run.run_hest_download:
    #     run_hest_download(cfg.download)

    # if cfg.run.run_segment:
    #     run_cell_segment(cfg.download, cfg.cellseg)

    # if cfg.run.run_overlay:
    #     run_overlay(cfg.download, cfg.cellseg)

    # if cfg.run.run_radiomics_extraction:
    #     run_radiomics_extraction(cfg.download, cfg.radiomics)

    # if cfg.run.run_statistics:
    #     run_statistics(cfg.download, cfg.statistics)


if __name__ == "__main__":
    run()



