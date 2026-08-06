#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "${script_dir}" rev-parse --show-toplevel)
source "${script_dir}/conda_helpers.sh"

ENV_NAME=${ENV_NAME:-nn-descent-mma-opt}
BASELINE_REF=${BASELINE_REF:-origin/main}
ROWS=${ROWS:-1000000}
REPEATS=${REPEATS:-5}
ITERATIONS=${ITERATIONS:-20}
GRAPH_DEGREE=${GRAPH_DEGREE:-64}
DIMS=${DIMS:-16,64,128,256,512,786,1024,1536}
GPU_ARCH=${GPU_ARCH:-auto}
RESULTS_ROOT=${RESULTS_ROOT:-/tmp/cuvs-nnd-mma-results}
SKIP_CANDIDATE_BUILD=0
SKIP_BASELINE_BUILD=0
KEEP_BASELINE_WORKTREE=0
RUN_SUCCEEDED=0

usage() {
  sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Usage:
#   run_benchmarks.sh [options]
#
# Options:
#   --env NAME                  Conda environment (default: nn-descent-mma-opt)
#   --baseline-ref REF          Baseline Git ref (default: origin/main)
#   --gpu-arch ARCH             CMake CUDA architecture or auto
#   --rows N                    Dataset rows
#   --dims CSV                  Dimensions, for example 64,256,1024
#   --repeats N                 Timed builds per case
#   --iterations N              NN-descent iterations
#   --degree N                  Output graph degree
#   --results-root DIR          Parent directory for the result bundle
#   --skip-candidate-build      Reuse cpp/build in this checkout
#   --skip-baseline-build       Reuse cpp/build in an existing --baseline-root
#   --baseline-root DIR         Use an existing main checkout instead of a temporary worktree
#   --keep-baseline-worktree    Do not remove the temporary main worktree
#   --smoke                     20k rows, 2 iterations, 1 repeat, dimensions 64,256
#   --help                      Show this help

BASELINE_ROOT=${BASELINE_ROOT:-}
while (($#)); do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --baseline-ref) BASELINE_REF="$2"; shift 2 ;;
    --gpu-arch) GPU_ARCH="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --dims) DIMS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --degree) GRAPH_DEGREE="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --baseline-root) BASELINE_ROOT="$2"; shift 2 ;;
    --skip-candidate-build) SKIP_CANDIDATE_BUILD=1; shift ;;
    --skip-baseline-build) SKIP_BASELINE_BUILD=1; shift ;;
    --keep-baseline-worktree) KEEP_BASELINE_WORKTREE=1; shift ;;
    --smoke) ROWS=20000; REPEATS=1; ITERATIONS=2; DIMS=64,256; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

conda_exe=$(find_conda_exe)
if ! conda_env_exists "${conda_exe}" "${ENV_NAME}"; then
  printf 'Conda environment %s does not exist. Run:\n' "${ENV_NAME}" >&2
  printf '  ENV_NAME=%q %q\n' "${ENV_NAME}" "${script_dir}/setup_conda_env.sh" >&2
  exit 2
fi
conda_prefix=$("${conda_exe}" run -n "${ENV_NAME}" bash -c 'printf %s "$CONDA_PREFIX"')
cuobjdump_exe=$("${conda_exe}" run -n "${ENV_NAME}" which cuobjdump)

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')
if [[ "${GPU_ARCH}" == auto ]]; then
  case "${compute_cap}" in
    9.0) GPU_ARCH=90a-real ;;
    10.0) GPU_ARCH=100a-real ;;
    12.0) GPU_ARCH=120-real ;;
    12.1) GPU_ARCH=121-real ;;
    *) GPU_ARCH="${compute_cap/./}-real" ;;
  esac
fi

candidate_sha=$(git -C "${repo_root}" rev-parse HEAD)
baseline_sha=$(git -C "${repo_root}" rev-parse "${BASELINE_REF}^{commit}")
run_id=$(printf '%s-%s-sm%s' \
  "$(date -u +%Y%m%dT%H%M%SZ)" "$(hostname -s)" "${compute_cap/./}" | tr -cs 'A-Za-z0-9._-' '-')
result_dir="${RESULTS_ROOT%/}/${run_id}"
mkdir -p "${result_dir}/bin"

created_worktree=0
worktree_parent=
cleanup() {
  if [[ "${created_worktree}" == 1 && "${KEEP_BASELINE_WORKTREE}" == 0 ]]; then
    if [[ "${RUN_SUCCEEDED}" == 1 ]]; then
      git -C "${repo_root}" worktree remove --force "${BASELINE_ROOT}" >/dev/null 2>&1 || true
      [[ -n "${worktree_parent}" ]] && rm -rf -- "${worktree_parent}"
    else
      printf "Validation failed; baseline worktree retained at %s\n" "${BASELINE_ROOT}" >&2
    fi
  fi
}
trap cleanup EXIT

if [[ -z "${BASELINE_ROOT}" ]]; then
  worktree_parent=$(mktemp -d "${TMPDIR:-/tmp}/cuvs-nnd-main.XXXXXX")
  BASELINE_ROOT="${worktree_parent}/main"
  git -C "${repo_root}" worktree add --detach "${BASELINE_ROOT}" "${BASELINE_REF}" \
    >"${result_dir}/main-worktree.log" 2>&1
  created_worktree=1
fi
BASELINE_ROOT=$(git -C "${BASELINE_ROOT}" rev-parse --show-toplevel)

build_tree() {
  local tree="$1"
  local log="$2"
  (
    cd "${tree}"
    "${conda_exe}" run -n "${ENV_NAME}" \
      ./build.sh libcuvs tests --gpu-arch="${GPU_ARCH}" --limit-tests=NEIGHBORS_TEST
  ) >"${log}" 2>&1
}

if [[ "${SKIP_CANDIDATE_BUILD}" == 0 ]]; then
  printf 'Building candidate %s for %s\n' "${candidate_sha}" "${GPU_ARCH}"
  build_tree "${repo_root}" "${result_dir}/candidate-build.log"
else
  printf 'Reusing candidate build at %s/cpp/build\n' "${repo_root}"
fi
if [[ "${SKIP_BASELINE_BUILD}" == 0 ]]; then
  printf 'Building main baseline %s for %s\n' "${baseline_sha}" "${GPU_ARCH}"
  build_tree "${BASELINE_ROOT}" "${result_dir}/main-build.log"
else
  printf 'Reusing baseline build at %s/cpp/build\n' "${BASELINE_ROOT}"
fi

compile_driver() {
  local tree="$1"
  local define="$2"
  local output="$3"
  local build_lib="${tree}/cpp/build"
  [[ -f "${build_lib}/libcuvs.so" ]] || {
    printf 'Missing built library: %s/libcuvs.so\n' "${build_lib}" >&2
    return 1
  }
  "${conda_exe}" run -n "${ENV_NAME}" nvcc -std=c++20 -O3 "-D${define}=1" \
    "${script_dir}/nn_descent_mma_bench.cu" -o "${output}" \
    -I"${repo_root}/cpp/build/_deps/cccl-src/libcudacxx/include" \
    -I"${repo_root}/cpp/build/_deps/cccl-src/thrust" \
    -I"${repo_root}/cpp/build/_deps/cccl-src/cub" \
    -I"${tree}/cpp/include" -I"${conda_prefix}/include" \
    -L"${build_lib}" -L"${conda_prefix}/lib" -lcuvs -lrmm -lrapids_logger \
    -Xcompiler=-fopenmp \
    -Xlinker=-rpath -Xlinker="${build_lib}" \
    -Xlinker=-rpath -Xlinker="${conda_prefix}/lib" \
    -Xlinker=-rpath-link -Xlinker="${conda_prefix}/lib"
}

printf 'Compiling benchmark drivers\n'
compile_driver "${repo_root}" NND_BENCH_CANDIDATE "${result_dir}/bin/candidate-bench"
compile_driver "${BASELINE_ROOT}" NND_BENCH_MAIN "${result_dir}/bin/main-bench"

run_driver() {
  local tree="$1"
  local binary="$2"
  local output="$3"
  local build_lib="${tree}/cpp/build"
  "${conda_exe}" run -n "${ENV_NAME}" env \
    "LD_LIBRARY_PATH=${build_lib}:${conda_prefix}/lib:${LD_LIBRARY_PATH:-}" \
    "${binary}" "${ROWS}" "${REPEATS}" "${ITERATIONS}" "${GRAPH_DEGREE}" "${DIMS}" \
    >"${output}" 2>"${output%.csv}.stderr.log"
}

printf 'Running candidate TF32 and FP16\n'
run_driver "${repo_root}" "${result_dir}/bin/candidate-bench" "${result_dir}/candidate.csv"
printf 'Running main FP32 and FP16\n'
run_driver "${BASELINE_ROOT}" "${result_dir}/bin/main-bench" "${result_dir}/main.csv"
{
  printf 'variant,mode,rows,dim,degree,iterations,median_ms,mean_ms,samples_ms\n'
  cat "${result_dir}/candidate.csv" "${result_dir}/main.csv"
} >"${result_dir}/combined.csv"

{
  printf 'run_id=%s\n' "${run_id}"
  printf 'utc_finished=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'hostname=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'candidate_sha=%s\n' "${candidate_sha}"
  printf 'baseline_ref=%s\n' "${BASELINE_REF}"
  printf 'baseline_sha=%s\n' "${baseline_sha}"
  printf 'compute_cap=%s\n' "${compute_cap}"
  printf 'gpu_arch=%s\n' "${GPU_ARCH}"
  printf 'env_name=%s\n' "${ENV_NAME}"
  printf 'conda_exe=%s\n' "${conda_exe}"
  printf 'rows=%s\nrepeats=%s\niterations=%s\ngraph_degree=%s\ndims=%s\n' \
    "${ROWS}" "${REPEATS}" "${ITERATIONS}" "${GRAPH_DEGREE}" "${DIMS}"
  printf 'candidate_dirty=%s\n' "$(git -C "${repo_root}" status --porcelain | wc -l)"
} >"${result_dir}/metadata.txt"

nvidia-smi -q >"${result_dir}/nvidia-smi-q.txt"
"${conda_exe}" run -n "${ENV_NAME}" nvcc --version >"${result_dir}/nvcc-version.txt"
"${conda_exe}" list -n "${ENV_NAME}" >"${result_dir}/conda-list.txt"
git -C "${repo_root}" diff >"${result_dir}/candidate-working-tree.diff"

collect_instructions() {
  local tree="$1"
  local prefix="$2"
  local library="${tree}/cpp/build/libcuvs.so"
  local symbols
  symbols=$("${cuobjdump_exe}" --list-text "${library}" 2>/dev/null \
    | grep -E "local_join_kernel_(tf32|wmma|simt)" \
    | sed -E "s/^.*x-//; s/\.sm_[^.]+\.elf\.bin$//" | sort -u | paste -sd, -)
  "${cuobjdump_exe}" --list-text "${library}" 2>/dev/null \
    | grep -E "local_join_kernel_(tf32|wmma|simt)" \
    >"${result_dir}/${prefix}-kernel-symbols.txt" || true
  if [[ -n "${symbols}" ]]; then
    "${cuobjdump_exe}" --dump-sass --function "${symbols}" "${library}" 2>/dev/null \
      | grep -Ei "WGMMA|TCGEN05|HMMA|MMA" >"${result_dir}/${prefix}-mma-instructions.txt" || true
  fi
  "${cuobjdump_exe}" --dump-resource-usage "${library}" 2>/dev/null | c++filt \
    | grep -B1 -A1 "local_join_kernel_" >"${result_dir}/${prefix}-resource-usage.txt" || true
}
collect_instructions "${repo_root}" candidate
collect_instructions "${BASELINE_ROOT}" main

summary_rows=$(mktemp "${TMPDIR:-/tmp}/nnd-summary.XXXXXX")
awk -F, '
  NR == 1 { next }
  {
    dim=$4; sub(/^dim=/, "", dim)
    median=$7; sub(/^median_ms=/, "", median)
    value[$1 SUBSEP $2 SUBSEP dim]=median
    dims[dim]=1
  }
  END {
    for (dim in dims) {
      tf32=value["candidate" SUBSEP "tf32" SUBSEP dim]
      cfp16=value["candidate" SUBSEP "fp16" SUBSEP dim]
      fp32=value["main" SUBSEP "fp32" SUBSEP dim]
      mfp16=value["main" SUBSEP "fp16" SUBSEP dim]
      if (tf32 != "" && fp32 != "" && cfp16 != "" && mfp16 != "") {
        printf "%s|%s|%s|%.3f|%s|%s|%.3f\n", dim, tf32, fp32, fp32/tf32, cfp16, mfp16, mfp16/cfp16
      }
    }
  }
' "${result_dir}/combined.csv" | sort -t'|' -k1,1n >"${summary_rows}"

{
  printf '# NN-descent MMA benchmark: %s\n\n' "${run_id}"
  printf -- '- Candidate: `%s`\n' "${candidate_sha}"
  printf -- '- Main baseline: `%s` (`%s`)\n' "${baseline_sha}" "${BASELINE_REF}"
  printf -- '- GPU: `%s`, compute capability `%s`, build target `%s`\n' \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')" \
    "${compute_cap}" "${GPU_ARCH}"
  printf -- '- Shape: rows `%s`; dimensions `%s`; degree `%s`; iterations `%s`; repeats `%s`\n\n' \
    "${ROWS}" "${DIMS}" "${GRAPH_DEGREE}" "${ITERATIONS}" "${REPEATS}"
  printf '| Dim | Candidate TF32 ms | Main FP32 ms | TF32 speedup | Candidate FP16 ms | Main FP16 ms | FP16 speedup |\n'
  printf '|---:|---:|---:|---:|---:|---:|---:|\n'
  while IFS='|' read -r dim tf32 fp32 tf32_speedup cfp16 mfp16 fp16_speedup; do
    printf '| %s | %s | %s | %.3fx | %s | %s | %.3fx |\n' \
      "${dim}" "${tf32}" "${fp32}" "${tf32_speedup}" "${cfp16}" "${mfp16}" "${fp16_speedup}"
  done <"${summary_rows}"
} >"${result_dir}/SUMMARY.md"
rm -f -- "${summary_rows}"

rm -rf -- "${result_dir}/bin"
tar -C "$(dirname -- "${result_dir}")" -czf "${result_dir}.tar.gz" "$(basename -- "${result_dir}")"
RUN_SUCCEEDED=1
printf 'RESULT_DIR=%s\n' "${result_dir}"
printf 'RESULT_ARCHIVE=%s.tar.gz\n' "${result_dir}"
printf 'SUMMARY=%s/SUMMARY.md\n' "${result_dir}"
