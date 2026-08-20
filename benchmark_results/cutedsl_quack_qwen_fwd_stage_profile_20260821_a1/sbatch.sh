#!/usr/bin/env bash
#SBATCH --job-name=mok-cute-fwd-stages
#SBATCH --account=dtcomp_b300
#SBATCH --qos=batch-short
#SBATCH --partition=b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=b300:8
#SBATCH --exclusive
#SBATCH --exclude=umb-b300-dp-185
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_results/cutedsl_quack_qwen_fwd_stage_profile_20260821_a1/slurm-%j.out
#SBATCH --error=benchmark_results/cutedsl_quack_qwen_fwd_stage_profile_20260821_a1/slurm-%j.err

set -Eeuo pipefail
umask 022

readonly BUNDLE=benchmark_results/cutedsl_quack_qwen_fwd_stage_profile_20260821_a1
readonly IMAGE='nvcr.io#nvidia/pytorch:26.05-py3'
readonly QUACK_SPEC='quack-kernels[cu13]==0.6.4'
readonly TK_COMMIT=1c3920d993404dd49a6d4c7267ea11d583bd5c68

inside_container() {
  local source_dir="$1" run_dir="$2" job_root="$3" source_commit="$4"
  local venv_dir="${job_root}/venv"
  export XDG_CACHE_HOME="${job_root}/cache/xdg"
  export PIP_CACHE_DIR="${job_root}/cache/pip"
  export TORCH_EXTENSIONS_DIR="${job_root}/cache/torch-extensions"
  export MOK_CUTEDSL_CACHE_ROOT="${job_root}/cache/cutedsl"
  export TMPDIR="${job_root}/tmp"
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
  export NCCL_IB_MERGE_NICS=0 OMP_NUM_THREADS=1
  unset PYTHONHOME PYTHONOPTIMIZE PYTHONPATH
  mkdir -p "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${MOK_CUTEDSL_CACHE_ROOT}" "${TMPDIR}"

  python -m venv --system-site-packages "${venv_dir}"
  "${venv_dir}/bin/python" -m pip install \
    --disable-pip-version-check --no-input "${QUACK_SPEC}"
  env MOK_ARCH=SM103 "${venv_dir}/bin/python" -m pip install \
    --disable-pip-version-check --no-input --no-deps --no-build-isolation \
    "${source_dir}"

  export MOK_SOURCE_COMMIT="${source_commit}"
  export MOK_STAGE_PROFILE_RESULT_JSON="${run_dir}/result.json"
  timeout --signal=TERM --kill-after=30s 24m \
    "${venv_dir}/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
    "${source_dir}/${BUNDLE}/run_profile.py"

  "${venv_dir}/bin/python" - "${run_dir}/result.json" <<'PY'
import json
import math
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["status"] == "PASS"
assert payload["correctness_gate"]["pass"] is True
assert payload["profile"]["samples"] == 5
metrics = payload["profile"]["metrics"]
assert set(("attributed_sum", "unattributed_gap", "total")) <= set(metrics)
for metric in metrics.values():
    assert len(metric["rank_max_samples_ms"]) == 5
    assert math.isfinite(metric["median_rank_max_ms"])
assert metrics["total"]["median_rank_max_ms"] > 0
print("validated_stage_profile=true")
PY
}

if [[ "${1:-}" == "--inside" ]]; then
  shift
  inside_container "$@"
  exit
fi

: "${SLURM_JOB_ID:?This script must run in a Slurm allocation}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${MOK_EXPECTED_HEAD:?Submit with MOK_EXPECTED_HEAD set to the exact commit}"
[[ "${MOK_EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]]

repo="$(realpath -- "${SLURM_SUBMIT_DIR}")"
readonly repo
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${MOK_EXPECTED_HEAD}" ]]
git -C "${repo}" cat-file -e "${MOK_EXPECTED_HEAD}:${BUNDLE}/run_profile.py"
git -C "${repo}" cat-file -e "${MOK_EXPECTED_HEAD}:${BUNDLE}/sbatch.sh"

readonly run_dir="${repo}/${BUNDLE}/runs/job-${SLURM_JOB_ID}"
mkdir -p "$(dirname "${run_dir}")"
mkdir "${run_dir}"
job_root="$(mktemp -d "/tmp/mok-cute-stages-${SLURM_JOB_ID}.XXXXXX")"
readonly job_root
case "${job_root}" in
  /tmp/mok-cute-stages-"${SLURM_JOB_ID}".*) ;;
  *) printf 'unsafe temporary path: %q\n' "${job_root}" >&2; exit 2 ;;
esac
trap 'rm -rf -- "${job_root}"' EXIT

readonly source_dir="${job_root}/source"
readonly tk_dir="${job_root}/tk"
mkdir -p "${source_dir}" "${tk_dir}" "${job_root}/cache" "${job_root}/tmp"
git -C "${repo}" archive "${MOK_EXPECTED_HEAD}" | tar -xf - -C "${source_dir}"
git init --quiet "${tk_dir}"
git -C "${tk_dir}" remote add origin https://github.com/HazyResearch/ThunderKittens.git
git -C "${tk_dir}" fetch --quiet --depth=1 origin "${TK_COMMIT}"
mkdir -p "${source_dir}/third_party/ThunderKittens"
git -C "${tk_dir}" archive FETCH_HEAD | \
  tar -xf - -C "${source_dir}/third_party/ThunderKittens"

printf 'job_id=%s\nsource_commit=%s\nnode=%s\n' \
  "${SLURM_JOB_ID}" "${MOK_EXPECTED_HEAD}" "${SLURMD_NODENAME:-unknown}" \
  > "${run_dir}/provenance.txt"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus=8 \
  --kill-on-bad-exit=1 --container-image="${IMAGE}" \
  --container-mounts="${repo}:${repo},${job_root}:${job_root}" \
  --container-workdir="${source_dir}" \
  bash "${source_dir}/${BUNDLE}/sbatch.sh" \
  --inside "${source_dir}" "${run_dir}" "${job_root}" "${MOK_EXPECTED_HEAD}"

[[ -s "${run_dir}/result.json" ]]
printf 'status=PASS\njob_id=%s\n' "${SLURM_JOB_ID}" > "${run_dir}/success.txt"
