"""Runtime GPU check -- the build has no device, so run this as a Job first.

Mirrors the verify block of build_cuml_env_nrp.sh.
"""
import sys

import cupy
import cuml
import numpy
import sklearn
import torch
import umap
import nemi

print("python       ", sys.version.split()[0])
print("numpy        ", numpy.__version__)
print("scikit-learn ", sklearn.__version__)
print("umap-learn   ", umap.__version__)
print("nemi         ", getattr(nemi, "__version__", "ok"))
print("torch cuda   ", torch.cuda.is_available())
print("cupy         ", cupy.__version__, "| jit:", int((cupy.arange(5) * 2).sum()))  # 20
print("cuml         ", cuml.__version__)
