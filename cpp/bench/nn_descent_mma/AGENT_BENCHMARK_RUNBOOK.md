# NN-descent MMA benchmark handoff for SM90 and SM100

## Objective

Run two matched comparisons on one otherwise-idle GPU:

1. `nn-descent-mma-opt` candidate TF32 versus `origin/main` FP32.
2. `nn-descent-mma-opt` candidate FP16 versus `origin/main` FP16.

The runner detects the installed GPU and chooses the build target automatically. On compute capability 9.0 it builds `90a-real`; on compute capability 10.0 it builds `100a-real`. Runtime dispatch inside cuVS then selects the appropriate NN-descent backend.

The default full run uses 1,000,000 rows, graph degree 64, 20 iterations, five timed repetitions, and dimensions `16,64,128,256,512,786,1024,1536`. Both revisions receive identical inputs and parameters.

## Clone and check out the candidate

```bash
git clone git@github.com:divyegala/cuvs.git
cd cuvs
git fetch origin main nn-descent-mma-opt
git switch --track origin/nn-descent-mma-opt
```

Verify that the checkout is clean:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/main
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
```

Do not benchmark while another process is using the GPU.

## Create the conda environment

The helper searches for conda under these roots, in order:

- `/raid/dgala`
- `/home/dgala`
- `/home/nfs/dgala`

It recognizes `miniforge3`, `mambaforge`, `miniconda3`, and `anaconda3`. Set `CONDA_EXE=/absolute/path/to/conda` if the installation is elsewhere.

Use the CUDA 12.9 all-development recipe and create the requested environment name:

```bash
CUDA_LINE=129 ENV_NAME=nn-descent-mma-opt \
  ./cpp/bench/nn_descent_mma/setup_conda_env.sh
```

If the environment already exists, the helper leaves it unchanged.

## Smoke test

Run this first. It builds the candidate and a detached `origin/main` worktree, compiles revision-specific benchmark drivers, and runs a small comparison:

```bash
ENV_NAME=nn-descent-mma-opt \
  ./cpp/bench/nn_descent_mma/run_benchmarks.sh --smoke
```

The final lines print `RESULT_DIR`, `RESULT_ARCHIVE`, and `SUMMARY`. Inspect `SUMMARY.md` and confirm all four modes are present.

## Full benchmark

```bash
ENV_NAME=nn-descent-mma-opt \
  ./cpp/bench/nn_descent_mma/run_benchmarks.sh
```

Useful overrides are available without editing files:

```bash
ENV_NAME=nn-descent-mma-opt \
  ROWS=1000000 REPEATS=7 ITERATIONS=20 GRAPH_DEGREE=64 \
  DIMS=64,128,256,512,1024,1536 \
  ./cpp/bench/nn_descent_mma/run_benchmarks.sh
```

Run the complete command at least once. Do not combine results from different GPUs, clocks, or driver configurations into one result directory.

## What the result bundle contains

The generated directory and `.tar.gz` archive include:

- `candidate.csv`: candidate TF32 and FP16 timings.
- `main.csv`: main FP32 and FP16 timings.
- `combined.csv`: all raw timing samples.
- `SUMMARY.md`: matched medians and speedups.
- `metadata.txt`: exact revisions, parameters, environment name, compute capability, and build target.
- `nvidia-smi-q.txt`, `nvcc-version.txt`, and `conda-list.txt`.
- Candidate and main build logs.
- MMA instruction extracts and CUDA resource-usage dumps for each library.
- A candidate working-tree diff, which should be empty on a clean checkout.

A speedup greater than 1.0 means the candidate is faster.

## Publish results for the orchestrating agent

Publish the full result directory to a unique branch on `divyegala/cuvs`:

```bash
./cpp/bench/nn_descent_mma/publish_results.sh /tmp/cuvs-nnd-mma-results/RUN_ID
```

The publisher creates a temporary worktree, commits the result bundle under `benchmark-results/RUN_ID`, and pushes a branch named `bench-results/RUN_ID`. It does not switch or modify the candidate checkout.

If you want to validate before pushing:

```bash
./cpp/bench/nn_descent_mma/publish_results.sh --dry-run /tmp/cuvs-nnd-mma-results/RUN_ID
```

Send the orchestrating agent these final values from the publisher:

```text
RESULT_BRANCH=bench-results/RUN_ID
RESULT_COMMIT=<commit>
```

If push authentication is unavailable, attach the generated `RESULT_ARCHIVE` to the task and report the candidate and main SHAs. Do not paste only the medians; the orchestrator needs the raw CSV and metadata.

## Orchestrator retrieval

Given a reported result branch, the orchestrator can retrieve it without checking it out:

```bash
git fetch origin RESULT_BRANCH
git show FETCH_HEAD:benchmark-results/RUN_ID/SUMMARY.md
git show FETCH_HEAD:benchmark-results/RUN_ID/combined.csv
```

The raw branch can then be compared with result branches from the other architecture.
