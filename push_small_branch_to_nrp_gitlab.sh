#!/usr/bin/env bash
# Push a notebook-free snapshot of main to the NRP GitLab project, which
# triggers the container image build.
#
# GitLab is only a build mirror: kaniko needs the Dockerfile plus the packaging
# and source, nothing else.  The notebooks carry embedded output images and
# their history alone exceeds GitLab's 128 MiB pack limit, so the build branch
# is an orphan (no history) holding only the paths listed in PATHS.
#
# Safe to run repeatedly.  Creates the branch and worktree on first run, then
# does incremental commits after that -- no force pushes.
set -euo pipefail

REPO="${REPO:-$HOME/PycharmProjects/DBOF_Representation_Learning}"
WORKTREE="${WORKTREE:-$HOME/nemi_fronts-build}"
BRANCH="nrp-build"
REMOTE="nrp"
SOURCE_BRANCH="main"
PIPELINES="https://gitlab.nrp-nautilus.io/jaketall/nemi_fronts/-/pipelines"

# Everything the image build touches.  Add to this if the Dockerfile starts
# COPYing something new.
PATHS=(Dockerfile .dockerignore .gitlab-ci.yml verify_gpu.py pyproject.toml src)

cd "$REPO"
git worktree prune

if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo ">> first run: creating orphan branch $BRANCH"
    git worktree add --detach "$WORKTREE" "$SOURCE_BRANCH"
    git -C "$WORKTREE" checkout --orphan "$BRANCH"
    git -C "$WORKTREE" reset -q                      # drop main's index; keep the files
    git -C "$WORKTREE" add -- "${PATHS[@]}"
    git -C "$WORKTREE" commit -q -m "Build context for NRP image"
elif [ ! -d "$WORKTREE" ]; then
    echo ">> attaching worktree at $WORKTREE"
    git worktree add "$WORKTREE" "$BRANCH"
fi

git -C "$WORKTREE" checkout -q "$BRANCH"
git -C "$WORKTREE" checkout "$SOURCE_BRANCH" -- "${PATHS[@]}"

if git -C "$WORKTREE" diff --cached --quiet && git -C "$WORKTREE" diff --quiet; then
    echo ">> build context unchanged since last push"
else
    git -C "$WORKTREE" add -A -- "${PATHS[@]}"
    git -C "$WORKTREE" commit -q -m "sync build context from $SOURCE_BRANCH $(git rev-parse --short "$SOURCE_BRANCH")"
    echo ">> committed $(git -C "$WORKTREE" rev-parse --short HEAD)"
fi

git -C "$WORKTREE" push "$REMOTE" "$BRANCH:main"
echo ">> pushed -- pipeline: $PIPELINES"