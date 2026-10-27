import os
from os.path import join as pjoin

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
)
irmas_path = pjoin(project_path, "data/IRMAS")
results_path = pjoin(project_path, "data/results")

classes = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

dataset_params = [
    "fs",
    "n_fft",
    "hop_length",
    "n_mels",
    "mono",
    "time_slice",
    "normalize",
]
network_params = ["base_network", "transfer", "mono", "epochs", "lr"]
