###########################################################################################
# Sync check T5: write a copy of a training config that resumes from an earlier run
#
# Sets Training.continue = 1 and Training.startfrom = <run name>, everything else unchanged.
# Only the json module is used, so it runs on a login node without the conda environment.
#
#   python3 make_resume_config.py <base config> <run name to resume> <output config>
###########################################################################################

import sys
import json

base_path, startfrom, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(base_path, "r") as f:
    config = json.load(f)
config["NeuralNetwork"]["Training"]["continue"] = 1
config["NeuralNetwork"]["Training"]["startfrom"] = startfrom
with open(out_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"wrote {out_path}: continue=1, startfrom={startfrom}")
