#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

# This script runs inside an already-held, serialized 8xB300 allocation.
# It never requests, reacquires, or releases resources.

readonly EXPECTED_TK_HEAD=1c3920d993404dd49a6d4c7267ea11d583bd5c68

die() {
  printf 'cutedsl-tiled-tma: %s\n' "$*" >&2
  exit 2
}

if (( $# != 1 )); then
  die 'usage: run_in_allocation.sh RUN_DIR'
fi

run_dir="$(realpath -- "$1")"
readonly run_dir
: "${SLURM_JOB_ID:?run inside the shared Slurm allocation}"
[[ -f "${run_dir}/source.tar" ]] || die 'source.tar is missing'
[[ -s "${run_dir}/source.tar.sha256" ]] || die 'source.tar.sha256 is missing'
[[ -s "${run_dir}/source_commit.txt" ]] || die 'source_commit.txt is missing'
[[ -s "${run_dir}/source_tree.txt" ]] || die 'source_tree.txt is missing'
source_commit="$(tr -d '[:space:]' < "${run_dir}/source_commit.txt")"
readonly source_commit
[[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || die 'invalid source commit'
source_tree="$(tr -d '[:space:]' < "${run_dir}/source_tree.txt")"
readonly source_tree
[[ "${source_tree}" =~ ^[0-9a-f]{40}$ ]] || die 'invalid source tree'

for command_name in git mktemp realpath sha256sum tar timeout; do
  command -v "${command_name}" >/dev/null || die "missing ${command_name}"
done
(cd "${run_dir}" && sha256sum -c source.tar.sha256)
for stale_path in \
  complete.txt terminal_status.txt SHA256SUMS \
  correctness/result.json correctness/status.txt \
  ab/result.json ab/status.txt ab/skipped.txt; do
  [[ ! -e "${run_dir}/${stale_path}" ]] || \
    die "refusing stale artifact: ${stale_path}"
done

job_root="$(mktemp -d "/tmp/mok-cutedsl-tiled-${SLURM_JOB_ID}.XXXXXX")"
readonly job_root
case "${job_root}" in
  /tmp/mok-cutedsl-tiled-"${SLURM_JOB_ID}".*) ;;
  *) die "unsafe temporary directory: ${job_root}" ;;
esac

cleanup() {
  local rc=$?
  trap - EXIT
  case "${job_root}" in
    /tmp/mok-cutedsl-tiled-"${SLURM_JOB_ID}".*) rm -rf -- "${job_root}" ;;
    *) printf 'refusing unsafe cleanup: %s\n' "${job_root}" >&2; rc=2 ;;
  esac
  exit "${rc}"
}
trap cleanup EXIT

readonly source_dir="${job_root}/source"
readonly tk_fetch_dir="${job_root}/thunderkittens-fetch"
mkdir -p "${source_dir}" "${tk_fetch_dir}" \
  "${job_root}/correctness" "${job_root}/ab" \
  "${run_dir}/correctness" "${run_dir}/ab"
tar -xf "${run_dir}/source.tar" -C "${source_dir}"

git init --quiet "${tk_fetch_dir}"
git -C "${tk_fetch_dir}" remote add origin \
  https://github.com/HazyResearch/ThunderKittens.git
git -C "${tk_fetch_dir}" fetch --quiet --depth=1 origin "${EXPECTED_TK_HEAD}"
[[ "$(git -C "${tk_fetch_dir}" rev-parse FETCH_HEAD)" == "${EXPECTED_TK_HEAD}" ]]
mkdir -p "${source_dir}/third_party/ThunderKittens"
git -C "${tk_fetch_dir}" archive "${EXPECTED_TK_HEAD}" | \
  tar -xf - -C "${source_dir}/third_party/ThunderKittens"

printf '%s\n' "${source_commit}" > "${run_dir}/runtime_source_commit.txt"
printf '%s\n' "${EXPECTED_TK_HEAD}" > "${run_dir}/thunderkittens_commit.txt"

run_phase() {
  local phase="$1"
  shift
  local phase_dir="${run_dir}/${phase}"
  local rc
  set +e
  "$@" > "${phase_dir}/stdout.log" 2> "${phase_dir}/stderr.log"
  rc=$?
  set -e
  printf 'phase=%s\nrc=%s\nend_utc=%s\n' \
    "${phase}" "${rc}" "$(date -u +%FT%TZ)" > "${phase_dir}/status.txt"
  return "${rc}"
}

readonly correctness_script="${source_dir}/benchmark_results/cutedsl_qwen_fwd_b300_correctness_20260821_a1/sbatch.sh"
readonly ab_script="${source_dir}/benchmark_results/cutedsl_quack_qwen_fwd_ab_20260821_a1/sbatch.sh"

if ! run_phase correctness \
  bash "${correctness_script}" --inside \
    "${source_dir}" "${run_dir}/correctness" "${job_root}/correctness"; then
  printf 'reason=correctness_failed\n' > "${run_dir}/ab/skipped.txt"
  printf 'status=FAIL\nphase=correctness\nend_utc=%s\n' \
    "$(date -u +%FT%TZ)" > "${run_dir}/terminal_status.txt"
  exit 1
fi

if ! run_phase ab \
  bash "${ab_script}" --inside \
    "${source_dir}" "${run_dir}/ab" "${job_root}/ab" "${source_commit}"; then
  printf 'status=FAIL\nphase=ab\nend_utc=%s\n' \
    "$(date -u +%FT%TZ)" > "${run_dir}/terminal_status.txt"
  exit 1
fi

printf 'status=PASS\nend_utc=%s\n' "$(date -u +%FT%TZ)" \
  > "${run_dir}/terminal_status.txt"
(
  cd "${run_dir}"
  sha256sum \
    source.tar source.tar.sha256 source_commit.txt source_tree.txt \
    runtime_source_commit.txt \
    thunderkittens_commit.txt \
    correctness/result.json correctness/stdout.log correctness/stderr.log \
    correctness/status.txt \
    ab/result.json ab/stdout.log ab/stderr.log ab/status.txt \
    terminal_status.txt > SHA256SUMS
)
printf 'status=PASS\nmanifest_sha256=%s\n' \
  "$(sha256sum "${run_dir}/SHA256SUMS" | awk '{print $1}')" \
  > "${run_dir}/complete.txt"
