#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "${script_dir}" rev-parse --show-toplevel)
source "${script_dir}/conda_helpers.sh"

ENV_NAME=${ENV_NAME:-nn-descent-mma-opt}
CUDA_LINE=${CUDA_LINE:-129}
machine_arch=$(uname -m)
case "${machine_arch}" in
  x86_64|aarch64) ;;
  *) printf 'Unsupported machine architecture: %s\n' "${machine_arch}" >&2; exit 2 ;;
esac

recipe="${repo_root}/conda/environments/all_cuda-${CUDA_LINE}_arch-${machine_arch}.yaml"
if [[ ! -f "${recipe}" ]]; then
  printf 'Conda recipe does not exist: %s\n' "${recipe}" >&2
  exit 2
fi

conda_exe=$(find_conda_exe)
if conda_env_exists "${conda_exe}" "${ENV_NAME}"; then
  printf 'Conda environment already exists: %s\n' "${ENV_NAME}"
  printf 'Use ENV_NAME=%s with the benchmark runner.\n' "${ENV_NAME}"
  exit 0
fi

printf 'Creating %s from %s using %s\n' "${ENV_NAME}" "${recipe}" "${conda_exe}"
"${conda_exe}" env create --name "${ENV_NAME}" --file "${recipe}"
printf 'Created conda environment: %s\n' "${ENV_NAME}"
