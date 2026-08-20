#!/usr/bin/env bash
#SBATCH --job-name=mok-cutedsl-quack-fwd-ab
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
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_results/cutedsl_quack_qwen_fwd_ab_20260821_a1/slurm-%j.out
#SBATCH --error=benchmark_results/cutedsl_quack_qwen_fwd_ab_20260821_a1/slurm-%j.err

set -Eeuo pipefail
umask 022

readonly BUNDLE=benchmark_results/cutedsl_quack_qwen_fwd_ab_20260821_a1
readonly IMAGE='nvcr.io#nvidia/pytorch:26.05-py3'
readonly QUACK_SPEC='quack-kernels[cu13]==0.6.4'
readonly EXPECTED_TK_HEAD=1c3920d993404dd49a6d4c7267ea11d583bd5c68

inside_container() {
  if (( $# != 4 )); then
    printf 'usage: sbatch.sh --inside SOURCE_DIR RUN_DIR JOB_ROOT SOURCE_COMMIT\n' >&2
    return 2
  fi
  local source_dir="$1"
  local run_dir="$2"
  local job_root="$3"
  local source_commit="$4"
  local venv_dir="${job_root}/venv"

  [[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]]
  [[ -f "${source_dir}/${BUNDLE}/run_ab.py" ]]
  [[ -f "${source_dir}/third_party/ThunderKittens/include/kittens.cuh" ]]
  export XDG_CACHE_HOME="${job_root}/cache/xdg"
  export PIP_CACHE_DIR="${job_root}/cache/pip"
  export TORCHINDUCTOR_CACHE_DIR="${job_root}/cache/torchinductor"
  export TRITON_CACHE_DIR="${job_root}/cache/triton"
  export CUDA_CACHE_PATH="${job_root}/cache/cuda"
  export TORCH_EXTENSIONS_DIR="${job_root}/cache/torch-extensions"
  export MOK_CUTEDSL_CACHE_ROOT="${job_root}/cache/cutedsl"
  export TMPDIR="${job_root}/tmp"
  export TMP="${TMPDIR}"
  export TEMP="${TMPDIR}"
  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export NCCL_DEBUG=WARN
  export NCCL_IB_MERGE_NICS=0
  export OMP_NUM_THREADS=1
  unset PYTHONHOME PYTHONOPTIMIZE PYTHONPATH
  mkdir -p "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" \
    "${MOK_CUTEDSL_CACHE_ROOT}" "${TMPDIR}"

  python -m venv --system-site-packages "${venv_dir}"
  "${venv_dir}/bin/python" -m pip install \
    --disable-pip-version-check --no-input "${QUACK_SPEC}"
  env MOK_ARCH=SM103 "${venv_dir}/bin/python" -m pip install \
    --disable-pip-version-check --no-input --no-deps --no-build-isolation \
    --force-reinstall "${source_dir}"
  "${venv_dir}/bin/python" - <<'PY'
import importlib.metadata
import torch

assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.6.2"
assert importlib.metadata.version("quack-kernels") == "0.6.4"
assert importlib.metadata.version("mixture-of-kittens") == "0.1.0"
assert torch.cuda.device_count() == 8
for index in range(8):
    assert torch.cuda.get_device_capability(index) == (10, 3)
    assert "B300" in torch.cuda.get_device_name(index).upper()
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY

  export MOK_SOURCE_COMMIT="${source_commit}"
  export MOK_AB_RESULT_JSON="${run_dir}/result.json"
  timeout --signal=TERM --kill-after=30s 50m \
    "${venv_dir}/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
    "${source_dir}/${BUNDLE}/run_ab.py"

  "${venv_dir}/bin/python" - \
    "${run_dir}/result.json" "${source_dir}/${BUNDLE}/run_ab.py" \
    "${source_commit}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

result_path = Path(sys.argv[1])
runner_path = Path(sys.argv[2])
source_commit = sys.argv[3]
payload = json.loads(result_path.read_text())
assert payload["status"] == "PASS"
assert payload["runtime_source_commit"] == source_commit
assert payload["runner_sha256"] == hashlib.sha256(runner_path.read_bytes()).hexdigest()
assert payload["comparison"] == "MoK CUDA FWD vs MoK CuTe DSL + QuACK FWD"
assert [case["tokens_per_rank"] for case in payload["cases"]] == [2048, 20480]
assert all(case["correctness"]["pass"] is True for case in payload["cases"])
assert payload["cases"][0]["correctness"]["mode"] == \
    "full_forward_output_against_pytorch_nccl_reference"
timing = payload["cases"][1]["timing"]
assert timing["warmups"] == 50 and timing["samples"] == 30
assert timing["backend_order"] == ["mok_cuda", "mok_cutedsl_quack"]
for backend in ("mok_cuda", "mok_cutedsl_quack"):
    values = timing[backend]
    assert len(values["rank_max_samples_ms"]) == 30
    assert all(math.isfinite(value) and value > 0 for value in values["rank_max_samples_ms"])
    assert math.isfinite(values["median_rank_max_ms"]) and values["median_rank_max_ms"] > 0
    assert math.isfinite(values["effective_tflops_per_gpu"]) and values["effective_tflops_per_gpu"] > 0
assert math.isfinite(timing["cutedsl_speedup_vs_cuda"])
assert timing["cutedsl_speedup_vs_cuda"] > 0
print("validated_result=true")
PY
}

if [[ "${1:-}" == "--inside" ]]; then
  shift
  inside_container "$@"
  exit
fi

: "${SLURM_JOB_ID:?This script must run in a Slurm allocation}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${MOK_EXPECTED_HEAD:?Submit with --export=ALL,MOK_EXPECTED_HEAD=$(git rev-parse HEAD)}"
[[ "${MOK_EXPECTED_HEAD}" =~ ^[0-9a-f]{40}$ ]]
for command_name in git mktemp realpath scontrol sha256sum srun tar; do
  command -v "${command_name}" >/dev/null
done

repo="$(realpath -- "${SLURM_SUBMIT_DIR}")"
readonly repo
readonly artifact_root="${repo}/${BUNDLE}/runs"
readonly run_dir="${artifact_root}/job-${SLURM_JOB_ID}"
[[ "$(realpath -- "$(git -C "${repo}" rev-parse --show-toplevel)")" == "${repo}" ]]
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${MOK_EXPECTED_HEAD}" ]]
git -C "${repo}" cat-file -e "${MOK_EXPECTED_HEAD}:${BUNDLE}/run_ab.py"
git -C "${repo}" cat-file -e "${MOK_EXPECTED_HEAD}:${BUNDLE}/sbatch.sh"
[[ "$(git -C "${repo}" ls-tree "${MOK_EXPECTED_HEAD}" third_party/ThunderKittens | awk '{print $3}')" == "${EXPECTED_TK_HEAD}" ]]
mkdir -p "${artifact_root}"
mkdir "${run_dir}"

job_root="$(mktemp -d "/tmp/mok-cutedsl-quack-ab-${SLURM_JOB_ID}.XXXXXX")"
readonly job_root
case "${job_root}" in
  /tmp/mok-cutedsl-quack-ab-"${SLURM_JOB_ID}".*) ;;
  *) printf 'Unsafe temporary path: %q\n' "${job_root}" >&2; exit 2 ;;
esac

cleanup() {
  local rc=$?
  trap - EXIT
  case "${job_root}" in
    /tmp/mok-cutedsl-quack-ab-"${SLURM_JOB_ID}".*) rm -rf -- "${job_root}" ;;
    *) printf 'Refusing unsafe cleanup path: %q\n' "${job_root}" >&2; rc=2 ;;
  esac
  if (( rc != 0 )); then
    printf 'status=FAILED\njob_id=%s\nexit_code=%s\nend_utc=%s\n' \
      "${SLURM_JOB_ID}" "${rc}" "$(date -u +%FT%TZ)" > "${run_dir}/failed.txt"
  fi
  exit "${rc}"
}
trap cleanup EXIT

readonly source_dir="${job_root}/source"
readonly tk_fetch_dir="${job_root}/thunderkittens-fetch"
mkdir -p "${source_dir}" "${job_root}/cache" "${job_root}/tmp"
git -C "${repo}" archive "${MOK_EXPECTED_HEAD}" | tar -xf - -C "${source_dir}"
git init --quiet "${tk_fetch_dir}"
git -C "${tk_fetch_dir}" remote add origin https://github.com/HazyResearch/ThunderKittens.git
git -C "${tk_fetch_dir}" fetch --quiet --depth=1 origin "${EXPECTED_TK_HEAD}"
[[ "$(git -C "${tk_fetch_dir}" rev-parse FETCH_HEAD)" == "${EXPECTED_TK_HEAD}" ]]
mkdir -p "${source_dir}/third_party/ThunderKittens"
git -C "${tk_fetch_dir}" archive "${EXPECTED_TK_HEAD}" | \
  tar -xf - -C "${source_dir}/third_party/ThunderKittens"

{
  printf 'job_id=%s\nstart_utc=%s\nnode=%s\naccount=%s\nqos=%s\npartition=%s\n' \
    "${SLURM_JOB_ID}" "$(date -u +%FT%TZ)" "${SLURMD_NODENAME:-unknown}" \
    "${SLURM_JOB_ACCOUNT:-unknown}" "${SLURM_JOB_QOS:-unknown}" \
    "${SLURM_JOB_PARTITION:-unknown}"
  printf 'source_commit=%s\nsource_tree=%s\nthunderkittens_commit=%s\ncontainer=%s\n' \
    "${MOK_EXPECTED_HEAD}" \
    "$(git -C "${repo}" show -s --format=%T "${MOK_EXPECTED_HEAD}")" \
    "${EXPECTED_TK_HEAD}" "${IMAGE}"
  printf 'runner_sha256=%s\nsbatch_sha256=%s\nquack_spec=%s\n' \
    "$(sha256sum "${source_dir}/${BUNDLE}/run_ab.py" | awk '{print $1}')" \
    "$(sha256sum "${source_dir}/${BUNDLE}/sbatch.sh" | awk '{print $1}')" \
    "${QUACK_SPEC}"
  scontrol show job --details --oneliner "${SLURM_JOB_ID}"
} > "${run_dir}/provenance.txt"

srun \
  --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus=8 \
  --kill-on-bad-exit=1 \
  --container-image="${IMAGE}" \
  --container-mounts="${repo}:${repo},${job_root}:${job_root}" \
  --container-workdir="${source_dir}" \
  bash "${source_dir}/${BUNDLE}/sbatch.sh" \
    --inside "${source_dir}" "${run_dir}" "${job_root}" "${MOK_EXPECTED_HEAD}"

[[ -s "${run_dir}/result.json" ]]
printf 'status=PASS\njob_id=%s\nend_utc=%s\nresult_sha256=%s\n' \
  "${SLURM_JOB_ID}" "$(date -u +%FT%TZ)" \
  "$(sha256sum "${run_dir}/result.json" | awk '{print $1}')" \
  > "${run_dir}/success.txt"
