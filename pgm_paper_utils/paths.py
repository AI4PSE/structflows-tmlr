"""Locations of results and figures. Override with environment variables."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# trained runs, evaluation CSVs and ablation outputs
RESULTS = os.environ.get("STRUCTFLOW_RESULTS", str(REPO / "results"))
# figures included by the paper
IMAGES = os.environ.get("STRUCTFLOW_IMAGES", str(REPO / "paper" / "images"))
