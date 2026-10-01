# building_jobs

How a script in this repo becomes a job running on NRP Nautilus.

Three systems, each with one role:

| | Role | Push here |
|---|---|---|
| GitHub (`origin`) | source of truth, full history | every code change |
| NRP GitLab (`nrp`) | build mirror -> CI -> registry | only when the image must change |
| Kubernetes (`kubectl`) | runs the work | every job |

They meet at one point: a job manifest names
`gitlab-registry.nrp-nautilus.io/jaketall/nemi_fronts:<tag>`.

## 1. Rebuild the image

Needed when the code the job runs changes, or when the conda/pip environment does.

```bash
git add -A && git commit -m "..." && ./push_small_branch_to_nrp_gitlab.sh
```

The script mirrors a notebook-free snapshot to an orphan branch on `nrp`, which
triggers kaniko and pushes two tags, `:<short-sha>` and `:latest`. Watch it under
CI/CD -> Pipelines.

The notebook-free mirror exists because notebooks carry embedded output images and
their history exceeds GitLab's 128 MiB pack limit.

Only the layers at or below the first change rebuild. Edits to `src/` or a root
script hit the `COPY` layer, leaving all five conda layers cached — minutes.
Editing a `mamba` line rebuilds the environment and re-pushes several GB.

**A new file at the repo root must be added to `PATHS` in the push script.**
Otherwise GitLab never receives it, `COPY . /opt/src/fronts` cannot include it, and
the job fails with `No such file or directory` after a full image pull.

## 2. Write the manifest

One file per job under `jobs/`. Four things matter:

```yaml
image: gitlab-registry.nrp-nautilus.io/jaketall/nemi_fronts:latest
command: ["python", "/opt/src/fronts/verify_gpu.py"]
envFrom:
  - secretRef:
      name: dbof-s3          # only jobs that read or write S3
resources:                   # requests must equal limits
  requests: {cpu: "2", memory: 8Gi, nvidia.com/gpu: "1"}
  limits:   {cpu: "2", memory: 8Gi, nvidia.com/gpu: "1"}
```

`python` resolves to the conda env because the Dockerfile puts
`${ENV_PREFIX}/bin` first on `PATH`; there is no `conda activate` anywhere.
`/opt/src/fronts` is where `COPY . /opt/src/fronts` lands the repo. Packages under
`src/` are also pip-installed, so `python -m <package>.<module>` works for anything
with a `__main__` guard.

`:latest` makes Kubernetes default `imagePullPolicy` to `Always`, so a pod picks up
the newest build without editing the manifest. A `<short-sha>` tag ties an artifact
to the code that produced it.

Jobs needing S3 require the `dbof-s3` secret to exist in the namespace
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). Without it the pod fails at startup
with `CreateContainerConfigError`. The bucket refuses anonymous access, so reads
need credentials too. Secrets are namespace-scoped and `sea-meets-the-stars` is
shared, so anyone in it can read them.

### GPU selection

`nodeAffinity` on `nvidia.com/gpu.product` restricts placement:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.product
              operator: In
              values: [NVIDIA-L40, NVIDIA-L40S, NVIDIA-RTX-A6000]
```

cuML 25.06 does not run on the cluster's Pascal cards (GTX 1080 Ti, TITAN Xp,
compute capability 6.1), and it fails only after the pod has pulled several GB.
The L40/L40S/A6000 tier carries 46-48 GB, which a cuML UMAP over the full patch
set needs. Widening to Ampere-or-newer consumer cards adds roughly 70 nodes at
24 GB. A100, H100, H200 and GH200 are capped at zero by namespace quota and are
requested as `nvidia.com/a100` rather than `nvidia.com/gpu`.

Verified working: L40S, driver 610.43.02, CUDA runtime 13.3.

## 3. Submit and watch

```bash
kubectl apply -f jobs/verify-gpu.yaml
kubectl get pod -l job-name=verify-gpu
kubectl logs job/verify-gpu
kubectl delete job verify-gpu
```

Budget 10-60 minutes queueing for a free GPU plus about 3 minutes for the image
pull on a cold node. A pod stays `Pending` indefinitely rather than failing.
`Completed` and `Failed` pods hold no hardware — CPU, memory and the GPU are
released when the container exits — but the pod object and its logs persist until
deleted.

A Job's pod template is immutable, so changing the image, affinity or resources
means deleting and recreating:

```bash
kubectl replace --force -f jobs/verify-gpu.yaml
```

Avoid `ttlSecondsAfterFinished` on jobs whose output you need to read; it deletes
the pod and its logs whether the job passed or failed.

## Environment pins

The image is built in staged conda layers with `conda-meta/pinned` holding
`cuml 25.06.*`, `numpy <2.5` and `scikit-learn 1.5.*`.

- `cuml` above 25.08 renames agglomerative's `n_neighbors` to `c`, which breaks
  NEMI. A bare install spec was not enough: a later `mamba install` bumped it to
  25.10.
- The layers are split because a single ~6 GB blob outlived the registry's auth
  token and failed with `UNAUTHORIZED` after 74 minutes.
- torch is the CPU build. GPU work is cuML and cupy; torch only converts tensors.
- `CONDA_OVERRIDE_CUDA=12.0` lets the solve succeed on a builder with no GPU,
  where the `__cuda` virtual package does not exist.
