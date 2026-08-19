# RAPIDS cuML + PyTorch image for NEMI jobs on NRP.
#
# Port of build_cuml_env_nrp.sh (NEMI repo).  The conda spec, the pins and the
# pip steps are identical; what changes is everything that only existed to
# survive JupyterHub: the $HOME cache/temp juggling, the pre-run cleanup, and
# the sitecustomize.py shim (an image can just set CUDA_PATH).
#
# CUDA userspace comes from conda, as in the script -- the driver is injected by
# the k8s device plugin, so no nvidia/cuda base image is needed.
#
# Build context is this repo, which is installed at the end (see FRONTS below).
FROM condaforge/miniforge3:26.3.2-3

ARG PY_VER=3.12
ARG ENV_PREFIX=/opt/conda_envs/main_cuml
ARG TARGET_TRIPLE=targets/x86_64-linux
ARG DBOF_URL=https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing.git
ARG DBOF_REF=cutouts_data_v2
ARG NEMI_URL=https://github.com/CompClimate/NEMI.git
ARG NEMI_REF=GPU-workflow

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Same network-resilience settings as the script; repodata fetches time out.
RUN conda config --set remote_connect_timeout_secs 60 \
 && conda config --set remote_read_timeout_secs 300 \
 && conda config --set remote_max_retries 8 \
 && conda config --set remote_backoff_factor 2

# ---------------------------------------------------------------------------
# The env.  Spec list copied verbatim from the script, including the pins:
#   numpy<2.5        -> numba (via cuml) ceiling
#   scikit-learn=1.5 -> cuml 25.06 sklearn-compat shim
#   cuml=25.06       -> GPU agglomerative takes `n_neighbors`, renamed to `c`
#                       after 25.08
#   cuda-*-dev/cccl/nvrtc -> headers cupy needs to JIT-compile kernels
# ---------------------------------------------------------------------------
# CONDA_PKGS_DIRS/TMPDIR are exported here rather than set as ENV: they exist
# only to keep the package cache and temp files out of the layer we keep, and an
# image-wide TMPDIR would point at a directory this RUN deletes.  libmamba does
# not create TMPDIR itself, hence the mkdir.
#
# CONDA_OVERRIDE_CUDA fakes the __cuda virtual package.  Conda derives __cuda
# from the host's NVIDIA driver; this builder has no GPU, so every CUDA build of
# pytorch and cupy is otherwise rejected as uninstallable.  (The shell script
# never needed this -- it ran on a JupyterHub GPU node.)  12.0 is the floor
# these packages ask for, which keeps the image runnable on the widest range of
# driver versions; raise it if the solve reports a higher __cuda requirement.
RUN mkdir -p /tmp/conda_pkgs /tmp/conda_tmp \
 && export CONDA_PKGS_DIRS=/tmp/conda_pkgs TMPDIR=/tmp/conda_tmp CONDA_OVERRIDE_CUDA=12.0 \
 && mamba create -p ${ENV_PREFIX} -y \
      -c rapidsai -c conda-forge -c nvidia \
      python=${PY_VER} "cuda-version=12.*" \
      cuml=25.06 cupy \
      cuda-cudart-dev cuda-cccl cuda-nvrtc \
      pytorch-gpu torchvision \
      "numpy<2.5" "scikit-learn=1.5.*" \
      zarr s3fs xarray dask ipykernel matplotlib \
 && conda clean --all -y \
 && rm -rf /tmp/conda_pkgs /tmp/conda_tmp

# Containers start without `conda activate`, so put the env on PATH instead.
# CUDA_PATH replaces the script's sitecustomize.py: cupy needs it to find the
# headers it JIT-compiles against, and the TARGETS dir is where they land.
ENV PATH=${ENV_PREFIX}/bin:${PATH} \
    CUDA_PATH=${ENV_PREFIX}/${TARGET_TRIPLE}

# dbof: --no-deps so it can't pull pip torch/numpy and re-tangle the CUDA stack.
RUN pip install --no-deps "dbof-in-native-grid @ git+${DBOF_URL}@${DBOF_REF}"

# umap-learn<0.5.7 for NEMI: 0.5.7+ calls sklearn 1.6's ensure_all_finite, but
# cuML pins scikit-learn to 1.5.x here.  (Also brings tqdm/pynndescent.)
RUN pip install einops timm==0.3.2 xmitgcm torchinfo "umap-learn<0.5.7"

# NEMI, for GPU cuML UMAP + clustering.  Base install only (no [gpu] extra):
# its cupy-cuda12x would clash with conda's cupy.
RUN git clone --branch ${NEMI_REF} --depth 1 ${NEMI_URL} /opt/git/NEMI \
 && pip install -e /opt/git/NEMI

# This repo (llc_cutout_dataloader + visualization).  --no-deps for the same
# reason as dbof: its pyproject pins torch==2.8.0, and letting pip honour that
# would replace conda's pytorch-gpu with a PyPI wheel.  Last layer, since this
# is the source that changes most often.
COPY . /opt/src/fronts
RUN pip install --no-deps /opt/src/fronts

# Usable as a Jupyter image too; --sys-prefix keeps the kernelspec in the env
# rather than in a $HOME that won't exist at runtime.
RUN python -m ipykernel install --sys-prefix --name main_cuml \
      --display-name "Python (cuML + torch)"

# Build-time check: imports only.  Anything touching a GPU lives in
# verify_gpu.py, which the builder has no device for -- run it as a Job.
RUN python -c "\
import importlib.util as u, numpy, sklearn, umap, torch, nemi, dbof, llc_cutout_dataloader; \
assert u.find_spec('cuml') and u.find_spec('cupy'), 'cuml/cupy not installed'; \
print('numpy', numpy.__version__, '| sklearn', sklearn.__version__, \
      '| umap', umap.__version__, '| torch', torch.__version__)"

WORKDIR /work
