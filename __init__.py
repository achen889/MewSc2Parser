import mew_utils


import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mew_log import *

log_config = None
with open("data/log/log_config.json", "r") as f:
    log_config = json.load(f)

if log_config:
    init_log(log_config)


import sc2reader