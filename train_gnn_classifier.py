import os
import runpy
import sys

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(here, "src"))
    script = os.path.join(here, "src", "train_gnn_classifier.py")
    runpy.run_path(script, run_name="__main__")


