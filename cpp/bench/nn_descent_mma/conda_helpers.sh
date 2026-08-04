#!/usr/bin/env bash

find_conda_exe() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return 0
  fi

  local root distribution candidate
  for root in /raid/dgala /home/dgala /home/nfs/dgala; do
    for distribution in miniforge3 mambaforge miniconda3 anaconda3; do
      candidate="${root}/${distribution}/bin/conda"
      if [[ -x "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  done

  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  printf '%s\n' "Unable to locate conda. Set CONDA_EXE explicitly." >&2
  return 1
}

conda_env_exists() {
  local conda_exe="$1"
  local env_name="$2"
  "${conda_exe}" env list | awk -v name="${env_name}" '$1 == name { found=1 } END { exit !found }'
}
