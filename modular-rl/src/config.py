import os

ENV_DIR = "environments"
XML_DIR = "environments/xmls"
BASE_MODULAR_ENV_PATH = "environments/ModularEnv.py"
# override with e.g. `export SWAT_DATA_DIR=/mnt/lime/sj2073/swat_data/results` on
# machines where the home directory is quota-limited shared storage but a larger
# (often machine-local) scratch mount is available -- see also wandb's own
# WANDB_DIR env var, which needs no code-side support here.
DATA_DIR = os.environ.get("SWAT_DATA_DIR", "results")
BUFFER_DIR = os.environ.get("SWAT_BUFFER_DIR", "buffers")
VIDEO_DIR = os.environ.get("SWAT_VIDEO_DIR", "results/videos")
VIDEO_RESOLUATION = (240, 240)

# ENV_DIR = "./environments"
# XML_DIR = "./environments/xmls"
# BASE_MODULAR_ENV_PATH = "./environments/ModularEnv.py"
# DATA_DIR = "./results"
# BUFFER_DIR = "./buffers"
# VIDEO_DIR = "./results/videos"
# VIDEO_RESOLUATION = (480, 480)
