#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "${script_dir}" rev-parse --show-toplevel)
REMOTE=${REMOTE:-origin}
DRY_RUN=0

usage() {
  printf 'Usage: %s [--remote NAME] [--branch NAME] [--dry-run] RESULT_DIR\n' "$0"
}

result_branch=${RESULT_BRANCH:-}
while (($#)); do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --branch) result_branch="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *) result_dir="$1"; shift ;;
  esac
done

if [[ -z "${result_dir:-}" ]]; then usage >&2; exit 2; fi
result_dir=$(realpath "${result_dir}")
for required in SUMMARY.md combined.csv metadata.txt nvidia-smi-q.txt; do
  [[ -f "${result_dir}/${required}" ]] || {
    printf 'Missing required result file: %s/%s\n' "${result_dir}" "${required}" >&2
    exit 2
  }
done

run_id=$(basename -- "${result_dir}")
if [[ -z "${result_branch}" ]]; then
  result_branch="bench-results/${run_id}"
fi
if git -C "${repo_root}" show-ref --verify --quiet "refs/heads/${result_branch}"; then
  printf 'Local branch already exists: %s\n' "${result_branch}" >&2
  exit 2
fi

printf 'RESULT_BRANCH=%s\n' "${result_branch}"
printf 'REMOTE=%s (%s)\n' "${REMOTE}" "$(git -C "${repo_root}" remote get-url "${REMOTE}")"
if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'Dry run: result bundle is valid and ready to publish.\n'
  exit 0
fi

publish_parent=$(mktemp -d "${TMPDIR:-/tmp}/cuvs-nnd-publish.XXXXXX")
publish_tree="${publish_parent}/worktree"
cleanup() {
  git -C "${repo_root}" worktree remove --force "${publish_tree}" >/dev/null 2>&1 || true
  rm -rf -- "${publish_parent}"
}
trap cleanup EXIT

git -C "${repo_root}" worktree add --detach "${publish_tree}" HEAD >/dev/null
git -C "${publish_tree}" switch -c "${result_branch}" >/dev/null
mkdir -p "${publish_tree}/benchmark-results"
cp -a "${result_dir}" "${publish_tree}/benchmark-results/${run_id}"
git -C "${publish_tree}" add "benchmark-results/${run_id}"
git -C "${publish_tree}" commit -m "Add NN-descent MMA benchmark results for ${run_id}"
git -C "${publish_tree}" push "${REMOTE}" "HEAD:refs/heads/${result_branch}"

result_commit=$(git -C "${publish_tree}" rev-parse HEAD)
printf 'RESULT_COMMIT=%s\n' "${result_commit}"
printf 'PUBLISHED_REF=%s/%s\n' "${REMOTE}" "${result_branch}"
printf 'Send RESULT_BRANCH and RESULT_COMMIT to the orchestrating agent.\n'
