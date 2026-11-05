"""Shared paths for all experiment scripts.

Set these environment variables to change where data and results live:
  EDGE_IDS_DATA  folder with the downloaded NetFlow v2 zip files (or extracted CSVs)
  EDGE_IDS_OUT   folder for the processed samples, splits and result CSV files
  EDGE_IDS_WORK  scratch folder for extracted CSVs and the edge-cost models
"""
import glob
import os
import zipfile

DATA = os.environ.get("EDGE_IDS_DATA", "data")
OUT = os.environ.get("EDGE_IDS_OUT", "processed")
WORK = os.environ.get("EDGE_IDS_WORK", "work")
for _d in (OUT, WORK):
    os.makedirs(_d, exist_ok=True)

# File names inside the University of Queensland zip archives.
KEYS = {
    "unsw": "NF-UNSW-NB15-v2.csv",
    "ton": "NF-ToN-IoT-v2.csv",
    "cic": "NF-CSE-CIC-IDS2018-v2.csv",
}


def get_csv(csv_name):
    """Return the path of an extracted dataset CSV, unzipping it if needed."""
    for root in (DATA, WORK):
        hits = glob.glob(os.path.join(root, "**", csv_name), recursive=True)
        if hits:
            return hits[0]
    for z in glob.glob(os.path.join(DATA, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith("/" + csv_name) or n == csv_name:
                    zf.extract(n, WORK)
                    return os.path.join(WORK, n)
    raise FileNotFoundError(f"{csv_name} not found in {DATA} or {WORK}")
