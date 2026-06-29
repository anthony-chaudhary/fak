#!/usr/bin/env python3
"""gcp_bench.py -- one-touch: provision a GCP GPU VM, benchmark fak on it, tear it down.

This is the provisioner + on-VM runner that turns "benchmark fak on GCP Blackwell"
into one command. It sits DOWNSTREAM of two peer-owned modules and never
re-implements them:

  * gcp_accel.py      -- the machine-type registry + Blackwell-first ladder
  * gcp_gpu_probe.py  -- the read-only "is this tier provisionable today" gate

The on-VM work is MODULAR: a single shared "driver" sets up the box once (apt, Go,
CUDA, the model fetch) and then runs one or more *engines* against the SAME GPU and
the SAME model, so the numbers are directly comparable. The engines today:

  * llama       -- llama.cpp(CUDA) llama-bench  (the baseline)
  * vllm        -- vLLM (PagedAttention + continuous batching) offline throughput
                   bench. A FIRST-CLASS SOTA-baseline peer to llama.cpp. vLLM loads
                   the HF (bf16) form of the SAME model -- it does not consume the
                   GGUF Q8 the llama/fak engines do -- so its row is a DIFFERENT
                   precision, disclosed in the normalized `precision` field. Its
                   number is an AGGREGATE continuous-batching throughput (vLLM's
                   actual strength), not the single-stream slice, also disclosed.
                   Opt-in (a ~5 GB install), so NOT pulled into the default `all`.
  * fak-cpu     -- fak's pure-Go Q8 engine via cmd/modelbench (CPU)
  * fak-cuda    -- fak's OWN engine on the CUDA backend via cmd/modelbench, f32
                   weights resident in VRAM (the un-narrowed device path). This is
                   the deliverable: fak's engine measured on a real datacenter GPU,
                   head-to-head vs llama.cpp on identical hardware.
  * fak-cuda-q8 -- the SAME engine + backend, but `-lean` so the GGUF Q8 weights go
                   resident as int8 codes + per-block f32 scales and the native Q8
                   device GEMV runs (the cuda backend advertises UploadDtype). This
                   is the APPLES-TO-APPLES decode row against llama.cpp's Q8_0:
                   decode is memory-bandwidth-bound, so streaming ~1 byte/weight
                   instead of 4 is the single largest decode lever (see
                   docs/benchmarks/H100-KERNEL-5X-ROADMAP.md). Opt-in until a green
                   on-H100 number witnesses the device Q8 GEMV -- then promote into `all`.
  * fak-cuda-tf32 -- the SAME engine + f32 device path, but with FAK_CUDA_TF32=1 so the f32
                   SGEMM runs on the Hopper/Ampere TENSOR CORES at TF32 input precision (F32
                   accumulate) instead of the FP32 CUDA cores. This is Lever 4 of the H100
                   roadmap -- the compute-bound PREFILL lever (f32 SGEMM leaves the tensor
                   cores idle). The weights stay f32; only the GEMM math narrows (mantissa
                   only), disclosed in the engine label. Opt-in for the same reason as
                   fak-cuda-q8: a deliberate side-by-side vs the pedantic-FP32 fak-cuda row.

`--engine {llama,vllm,fak-cpu,fak-cuda,fak-cuda-q8,fak-cuda-tf32,fak,all}` selects which run (default: all;
vLLM is opt-in via `--engine vllm` or a comma list like `llama,vllm`). Every
engine writes a normalized {prefill,decode}_tok_per_sec row; the driver folds them
into ONE result.json (schema fak.gcp-vm-bench.v2) with an `engines` map plus a
back-compatible top-level headline (the llama baseline), and that result folds into
the cross-machine benchmark catalog unchanged.

The fak source is shipped to the VM by SCP of a tarball of the LOCAL working tree
(the repo is private; this also benches the exact code under test, including
uncommitted changes), never a git clone -- no repo auth on the VM, ever.

Flow (each step is logged and gated):

  1. preflight    -- resolve the tier (probe the ladder unless one is pinned),
                     confirm auth is live; STOP with the exact fix if not.
  2. provision    -- `gcloud compute instances create` with the tier's
                     accelerator, a CUDA Deep-Learning-VM image, and the NVIDIA
                     driver install. A trivial startup-script just marks boot.
  3. ship+run     -- SCP a tarball of fak/ + the rendered driver to the VM, SSH in,
                     run the driver (build+bench every selected engine), read JSON.
  4. collect      -- read result.json back, fold it into the benchmark catalog under
                     experiments/benchmark/runs/by-machine/<machine-id>/, register
                     the machine specs.
  5. teardown     -- ALWAYS delete the instance (and disk) on the way out, even on
                     failure or Ctrl-C. A leaked Blackwell VM is real money.

SAFETY: teardown is in a finally block AND idempotent (a second delete is a
no-op). `--keep` skips teardown for debugging but prints a loud warning + the
exact delete command. `--dry-run` prints every gcloud command without running
any, so the whole path is reviewable offline and in CI.

Usage:
  python tools/gcp_bench.py --dry-run                 # print the plan, touch nothing
  python tools/gcp_bench.py --tier g2-l4              # L4: all engines head-to-head
  python tools/gcp_bench.py --tier a4-b200 --blackwell  # the flagship Blackwell run
  python tools/gcp_bench.py --proof --engine llama    # cheapest L4, just the baseline
  python tools/gcp_bench.py --keep --tier g2-l4       # leave the VM up to debug
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import datetime as _dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from typing import Optional

import gcp_accel
import gcp_gpu_probe as probe


SCHEMA = "fak.gcp-bench.v1"
ROOT = Path(__file__).resolve().parents[1]
# The Go module source root. In the private monorepo it was a fak/ subdir; in the
# public tree the module IS the repo root. Auto-detect so the tarball, the on-VM
# `$SRC/go.mod` guard, and the runs/machines dirs all resolve on either layout.
FAK_SRC = (ROOT / "fak") if (ROOT / "fak" / "go.mod").is_file() else ROOT
RUNS_DIR = FAK_SRC / "experiments" / "benchmark" / "runs" / "by-machine"
MACHINES_DIR = FAK_SRC / "experiments" / "benchmark" / "machines"

# The model the on-VM bench runs. A small, fast-to-fetch GGUF that exercises the
# real decode path on the datacenter GPU without a multi-hundred-GB download --
# the point is the HARDWARE number + a proven pipeline, not a 27B run (that is the
# DGX ladder's job). Operator can override with --hf-repo/--hf-file.
DEFAULT_HF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_HF_FILE = "qwen2.5-3b-instruct-q8_0.gguf"


@dataclass(frozen=True)
class Engine:
    """One way to benchmark on the VM. `needs_cuda` engines are skipped (with a
    recorded note) when the box has no usable GPU build path."""

    key: str
    label: str
    needs_cuda: bool


# The engine registry. ENGINE_ORDER is the canonical run/headline order (baselines
# first, then fak's own engine; llama is the comparable headline baseline). vLLM is a
# first-class SOTA-baseline peer to llama.cpp and orders right after it, so a comma
# list like `--engine llama,vllm` runs the engine head-to-head in a stable order.
ENGINES: dict[str, Engine] = {
    "llama": Engine("llama", "llama.cpp (CUDA) baseline", needs_cuda=True),
    "vllm": Engine("vllm", "vLLM (PagedAttention, continuous batching) baseline", needs_cuda=True),
    "fak-cpu": Engine("fak-cpu", "fak pure-Go Q8 engine (CPU)", needs_cuda=False),
    "fak-cuda": Engine("fak-cuda", "fak engine on the CUDA backend (f32)", needs_cuda=True),
    "fak-cuda-q8": Engine("fak-cuda-q8", "fak engine on the CUDA backend (Q8 device GEMV; apples-to-apples vs llama.cpp Q8_0)", needs_cuda=True),
    "fak-cuda-tf32": Engine("fak-cuda-tf32", "fak engine on the CUDA backend (f32 weights, TF32 tensor-core SGEMM math; prefill tensor-core lever)", needs_cuda=True),
}
ENGINE_ORDER = ["llama", "vllm", "fak-cpu", "fak-cuda", "fak-cuda-q8", "fak-cuda-tf32"]
# `all` stays the CURATED fak-vs-llama.cpp default and deliberately EXCLUDES vLLM:
# vLLM is a ~5 GB install on a different (server) serving paradigm, so pulling it
# into every default run would change the cost/behaviour of existing benches. vLLM is
# therefore opt-in -- select it explicitly (`--engine vllm`) or in a comma list.
# fak-cuda-q8 is likewise opt-in: the f32 fak-cuda row is the witnessed device path
# today; the Q8 device GEMV has off-GPU cosine witnesses but no on-H100 number yet, so
# it is selected explicitly (`--engine fak-cuda,fak-cuda-q8`) and PROMOTED into `all`
# only once a green Hopper run witnesses it (docs/benchmarks/H100-KERNEL-5X-ROADMAP.md).
# fak-cuda-tf32 (Lever 4: the same f32 device path with TF32 tensor-core SGEMM math, via
# FAK_CUDA_TF32=1) is opt-in for the same reason — it targets the compute-bound PREFILL row
# and changes the f32 GEMM numerics (mantissa-only), so it stays a deliberate side-by-side
# against the pedantic-FP32 fak-cuda row until a green Hopper run witnesses its prefill gain.
DEFAULT_ALL = ["llama", "fak-cpu", "fak-cuda"]


def resolve_engines(spec: str) -> list[str]:
    """Map a --engine value to an ordered, de-duplicated list of engine keys."""
    if spec == "all":
        return list(DEFAULT_ALL)
    if spec == "fak":
        return ["fak-cpu", "fak-cuda"]
    if spec in ENGINES:
        return [spec]
    # comma list
    keys = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [k for k in keys if k not in ENGINES]
    if bad:
        raise ValueError(f"unknown engine(s): {bad}; valid: {list(ENGINES)} (or 'fak'/'all')")
    return [k for k in ENGINE_ORDER if k in keys]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str) -> None:
    print(f"[gcp-bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _gcloud_exe() -> str:
    return shutil.which("gcloud") or shutil.which("gcloud.cmd") or "gcloud"


def write_lf(path: Path, body: str) -> None:
    """Write a shell script with LF line endings even on Windows.

    Path.write_text / open() default to newline translation, so on this Windows host
    a rendered bash script lands as CRLF -- which the VM's bash rejects outright
    ('set: pipefail\\r: invalid option', 'syntax error near `{\\r`'). Force LF so the
    SAME bytes run on the Linux VM as `bash -n` sees here.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)


class Runner:
    """Thin gcloud runner that honours --dry-run uniformly."""

    def __init__(self, dry_run: bool, project: Optional[str], account: Optional[str]):
        self.dry_run = dry_run
        self.project = project
        self.account = account
        self.exe = _gcloud_exe()

    def _base(self) -> list[str]:
        cmd = [self.exe]
        if self.project:
            cmd += ["--project", self.project]
        if self.account:
            cmd += ["--account", self.account]
        return cmd

    def run(self, args: list[str], *, capture: bool = False, timeout: int = 600,
            check: bool = True) -> subprocess.CompletedProcess:
        cmd = self._base() + args
        printable = " ".join(shlex.quote(c) for c in cmd)
        if self.dry_run:
            log(f"DRY-RUN gcloud: {printable}")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        log(f"gcloud: {printable}")
        try:
            proc = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            # A hung gcloud (SSH/scp stall) must NOT escape as TimeoutExpired: that would
            # abort a retry loop (wait_for_ssh) on the first stall, or skip the result.json
            # read. Treat a timeout as a failed call -- raise on check=True, else return
            # rc=124 so the caller retries / falls through to read whatever landed.
            msg = f"gcloud timed out after {timeout}s: {' '.join(args[:3])}..."
            if check:
                raise RuntimeError(msg) from e
            log(f"WARNING: {msg}")
            return subprocess.CompletedProcess(cmd, 124, e.stdout or "", e.stderr or "")
        if check and proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() if capture else ""
            raise RuntimeError(f"gcloud failed (rc={proc.returncode}): {' '.join(args[:3])}...\n{err}")
        return proc

    def scp(self, local: str, name: str, remote: str, zone: str,
            *, to_vm: bool = True, timeout: int = 1200) -> subprocess.CompletedProcess:
        """scp a file to/from the VM over IAP. No -o flags (Windows gcloud shells
        out to plink/scp, which rejects OpenSSH-style options)."""
        ep = f"{name}:{remote}"
        src, dst = (local, ep) if to_vm else (ep, local)
        return self.run(
            ["compute", "scp", "--tunnel-through-iap", src, dst, f"--zone={zone}"],
            capture=True, timeout=timeout, check=True,
        )


# ----------------------------------------------------------------------------
# On-VM scripts. Two pieces:
#   * the boot startup-script (trivial: just mark ready; the DLVM image + the
#     install-nvidia-driver metadata handle the driver),
#   * the driver: one shared setup + N engine fragments + a combiner. Built with
#     @@PLACEHOLDER@@ substitution (NOT an f-string) so the heavy bash/python --
#     full of { } and $ -- needs no escaping.
# ----------------------------------------------------------------------------

def render_setup_script() -> str:
    """The boot startup-script. The real work is the SCP'd driver; this only marks
    the box ready so wait_for_ssh + the driver run deterministically over SSH."""
    return (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "mkdir -p /opt/gcp-bench\n"
        'echo "boot-ready $(date -u +%FT%TZ)" > /opt/gcp-bench/boot.marker\n'
    )


# Small python helpers written to the VM once, then invoked with argv. Kept out of
# the bash so neither bash nor a heredoc has to escape their quotes/braces.
_PY_NORM_LLAMA = r'''import json,sys
raw=json.load(open(sys.argv[1])); gpus=sys.argv[2]; model=sys.argv[3]; out=sys.argv[4]
def pick(rows,tag):
    for r in (rows if isinstance(rows,list) else []):
        if r.get("n_prompt") and tag=="pp": return r.get("avg_ts")
        if r.get("n_gen") and tag=="tg": return r.get("avg_ts")
    return None
json.dump({"engine":"llama","ok":True,"backend":"llama.cpp CUDA","precision":"Q8_0",
  "prefill_tok_per_sec":pick(raw,"pp"),"decode_tok_per_sec":pick(raw,"tg"),
  "gpus":gpus,"model":model,"raw":raw,"raw_artifact":"llama-bench.json"},
  open(out,"w"),indent=2)
'''

_PY_NORM_FAK = r'''import json,sys
key=sys.argv[1]; rep=json.load(open(sys.argv[2])); out=sys.argv[3]
pf=None
for p in (rep.get("prefill") or []):
    if p.get("tokens")==512: pf=p.get("tok_per_sec")
if pf is None and rep.get("prefill"): pf=rep["prefill"][-1].get("tok_per_sec")
dec=(rep.get("decode") or {}).get("tok_per_sec")
json.dump({"engine":key,"ok":True,"backend":rep.get("engine"),"precision":rep.get("precision"),
  "prefill_tok_per_sec":pf,"decode_tok_per_sec":dec,"workers":rep.get("workers"),
  "load_ms":rep.get("load_ms"),"raw":rep,"raw_artifact":sys.argv[2].split("/")[-1]},
  open(out,"w"),indent=2)
'''

_PY_NORM_VLLM = r'''import json,sys
raw=json.load(open(sys.argv[1])); model=sys.argv[2]; out=sys.argv[3]
def g(*keys):
    for k in keys:
        v=raw.get(k)
        if isinstance(v,(int,float)) and not isinstance(v,bool): return v
    return None
# vLLM `bench throughput` is an AGGREGATE continuous-batching number (vLLM's actual
# strength), NOT a single-stream slice. Prefer the OUTPUT-token throughput; fall back
# to total_tokens/elapsed (labelled). If no known key is present, ok=False -- the
# combiner records a diagnosed miss, NEVER a fabricated number.
decode=g("output_throughput","output_token_throughput","output_tokens_per_second","output_toks_per_s","tokens_per_second")
note="vLLM AGGREGATE continuous-batching output tok/s"
if decode is None:
    tot=g("total_num_tokens","num_tokens","total_tokens"); el=g("elapsed_time","duration","elapsed_s","elapsed")
    if tot and el and el>0:
        decode=tot/el; note="vLLM AGGREGATE total tok/s (total_num_tokens/elapsed fallback)"
json.dump({"engine":"vllm","ok":decode is not None,
  "backend":"vLLM (PagedAttention, continuous batching)","precision":"bf16",
  "prefill_tok_per_sec":None,"decode_tok_per_sec":decode,
  "note":note+"; HF bf16 weights of the SAME model, NOT the GGUF Q8 the llama/fak rows use, and NOT single-stream -- vLLM's edge is concurrency",
  "model":model,"raw":raw,"raw_artifact":sys.argv[1].split("/")[-1]},
  open(out,"w"),indent=2)
'''

_PY_DIAG = r'''import json,sys,subprocess
key,logf,out=sys.argv[1],sys.argv[2],sys.argv[3]
def tail(p,n=60):
    try: return subprocess.run(["tail","-n",str(n),p],capture_output=True,text=True).stdout
    except Exception as e: return "tail failed: "+str(e)
json.dump({"engine":key,"ok":False,"error":key+" failed on VM","log_tail":tail(logf)},
  open(out,"w"),indent=2)
'''

_PY_COMBINE = r'''import json,sys,glob,os
work,gpus,model,arch=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
engines={}
for f in sorted(glob.glob(os.path.join(work,"engine-*.json"))):
    k=os.path.basename(f)[len("engine-"):-len(".json")]
    try: engines[k]=json.load(open(f))
    except Exception as e: engines[k]={"engine":k,"ok":False,"error":"unreadable: "+str(e)}
head=None
for k in (["llama"]+list(engines.keys())):
    e=engines.get(k)
    if e and e.get("ok") and e.get("decode_tok_per_sec") is not None: head=e; break
out={"schema":"fak.gcp-vm-bench.v2","gpus":gpus,"model":model,"arch":arch,"engines":engines}
n_ok=sum(1 for e in engines.values() if e and e.get("ok"))
out["ok"]=n_ok>0
if not engines:
    out["error"]="no engine produced a result"
elif n_ok==0:
    out["error"]="all %d engine(s) failed" % len(engines)
if head:
    out["prefill_tok_per_sec"]=head.get("prefill_tok_per_sec")
    out["decode_tok_per_sec"]=head.get("decode_tok_per_sec")
    out["headline_engine"]=head.get("engine")
json.dump(out,open(os.path.join(work,"result.json"),"w"),indent=2)
print("ENGINES SUMMARY:")
for k,v in engines.items():
    print("  %s: ok=%s prefill=%s decode=%s" % (k,v.get("ok"),v.get("prefill_tok_per_sec"),v.get("decode_tok_per_sec")))
'''


# Per-engine bash function bodies. Each builds + runs its engine and writes a
# normalized $WORK/engine-<key>.json on success; the run_one wrapper writes a
# diagnosis on failure. $WORK,$SRC,$MODEL,$MODEL_FILE,$GPUS,$CC are set by the
# shared preamble.
_ENGINE_BASH = {
    "llama": r'''engine_llama(){
  cd "$WORK"
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_CC"
  cmake --build build --config Release -j"$(nproc)" --target llama-bench
  ./build/bin/llama-bench -m "$MODEL" -ngl 99 -p 512 -n 128 -o json > "$WORK/llama-bench.json"
  python3 "$WORK/norm_llama.py" "$WORK/llama-bench.json" "$GPUS" "$MODEL_FILE" "$WORK/engine-llama.json"
}''',
    # vLLM serves the HF (bf16) form of the SAME model (it does not load the GGUF Q8),
    # via its offline `bench throughput` (no server needed). The number is AGGREGATE
    # continuous-batching throughput -- vLLM's actual strength -- recorded honestly as
    # bf16 + aggregate in the normalized row. A pip/build/arch mismatch fails THIS
    # engine only (run_one diagnoses it); it never fabricates or blocks the others.
    "vllm": r'''engine_vllm(){
  cd "$WORK"
  VLLM_MODEL="${MODEL_REPO%-GGUF}"
  python3 -m pip install -q -U vllm >/dev/null 2>&1 || pip3 install -q -U vllm >/dev/null 2>&1 || pip install -q -U vllm
  # Try the modern `vllm bench throughput` CLI, then the module entrypoint, writing a
  # JSON the normalizer reads. input/output lengths mirror the llama-bench pp512/tg128.
  vllm bench throughput --model "$VLLM_MODEL" --input-len 512 --output-len 128 \
      --num-prompts 64 --dtype bfloat16 --output-json "$WORK/vllm-throughput.json" \
    || python3 -m vllm.entrypoints.cli.main bench throughput --model "$VLLM_MODEL" \
        --input-len 512 --output-len 128 --num-prompts 64 --dtype bfloat16 \
        --output-json "$WORK/vllm-throughput.json"
  python3 "$WORK/norm_vllm.py" "$WORK/vllm-throughput.json" "$VLLM_MODEL" "$WORK/engine-vllm.json"
}''',
    "fak-cpu": r'''engine_fak_cpu(){
  cd "$SRC"
  go build -o "$WORK/bin/modelbench" ./cmd/modelbench
  "$WORK/bin/modelbench" -gguf "$MODEL" -lean \
      -prefill-sizes 512 -prefill-reps 5 -decode-steps 128 -decode-reps 5 -decode-prompt 16 \
      -out "$WORK/fak-cpu-report.json"
  python3 "$WORK/norm_fak.py" fak-cpu "$WORK/fak-cpu-report.json" "$WORK/engine-fak-cpu.json"
}''',
    # fak-cuda runs f32 (no -lean/-quant): the un-narrowed device path -- f32 weights
    # resident in VRAM (uploaded once, cached). That streams 4 bytes/weight where the
    # llama/fak-cpu rows stream ~1 (Q8); decode is memory-bandwidth-bound, so the
    # apples-to-apples Q8 device row is the fak-cuda-q8 engine below (the cuda backend
    # DOES advertise UploadDtype). -require-non-reference fails loudly if the run did NOT
    # actually land on the GPU, so a green number can't be a silent CPU fallback.
    "fak-cuda": r'''engine_fak_cuda(){
  cd "$SRC"
  export CUDA_HOME=/usr/local/cuda
  FAK_CUDA_ARCH="$CUDA_CC" bash internal/compute/build_cuda.sh build
  export CGO_ENABLED=1
  export CGO_CFLAGS="-I/usr/local/cuda/include"
  export CGO_LDFLAGS="-L$SRC/internal/compute -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64"
  go build -tags cuda -o "$WORK/bin/modelbench-cuda" ./cmd/modelbench
  LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    "$WORK/bin/modelbench-cuda" -gguf "$MODEL" -backend cuda -require-non-reference \
      -prefill-sizes 512 -prefill-reps 5 -decode-steps 128 -decode-reps 5 -decode-prompt 16 \
      -out "$WORK/fak-cuda-report.json"
  python3 "$WORK/norm_fak.py" fak-cuda "$WORK/fak-cuda-report.json" "$WORK/engine-fak-cuda.json"
}''',
    # fak-cuda-q8 is fak-cuda + `-lean`: the GGUF Q8 weights stay resident as int8 codes +
    # per-block f32 scales and the native Q8 device GEMV runs (no f32 dequant round-trip), so
    # decode streams ~1 byte/weight instead of 4 -- the apples-to-apples row vs llama.cpp Q8_0.
    # It REUSES the modelbench-cuda binary fak-cuda already built (guarded), so selecting both
    # engines compiles the cuda backend once; run alone, it builds the binary itself.
    "fak-cuda-q8": r'''engine_fak_cuda_q8(){
  cd "$SRC"
  if [ ! -x "$WORK/bin/modelbench-cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    FAK_CUDA_ARCH="$CUDA_CC" bash internal/compute/build_cuda.sh build
    export CGO_ENABLED=1
    export CGO_CFLAGS="-I/usr/local/cuda/include"
    export CGO_LDFLAGS="-L$SRC/internal/compute -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64"
    go build -tags cuda -o "$WORK/bin/modelbench-cuda" ./cmd/modelbench
  fi
  LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    "$WORK/bin/modelbench-cuda" -gguf "$MODEL" -lean -backend cuda -require-non-reference \
      -prefill-sizes 512 -prefill-reps 5 -decode-steps 128 -decode-reps 5 -decode-prompt 16 \
      -out "$WORK/fak-cuda-q8-report.json"
  python3 "$WORK/norm_fak.py" fak-cuda-q8 "$WORK/fak-cuda-q8-report.json" "$WORK/engine-fak-cuda-q8.json"
}''',
    # fak-cuda-tf32 is fak-cuda's SAME f32 device path (no -lean: f32 weights resident in VRAM)
    # but with FAK_CUDA_TF32=1, which routes the f32 SGEMM (the prefill projections) through the
    # Hopper/Ampere TENSOR CORES at TF32 input precision (F32 accumulate) instead of the FP32 CUDA
    # cores — Lever 4 of the H100 roadmap, the compute-bound PREFILL lever. It REUSES the
    # modelbench-cuda binary fak-cuda already built (guarded), so selecting both compiles the cuda
    # backend once; run alone it builds the binary itself. -require-non-reference keeps a silent CPU
    # fallback from masquerading as a TF32 number. The TF32 math is disclosed in the engine label;
    # the resident weights are still f32, so the report's precision field stays "f32".
    "fak-cuda-tf32": r'''engine_fak_cuda_tf32(){
  cd "$SRC"
  if [ ! -x "$WORK/bin/modelbench-cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    FAK_CUDA_ARCH="$CUDA_CC" bash internal/compute/build_cuda.sh build
    export CGO_ENABLED=1
    export CGO_CFLAGS="-I/usr/local/cuda/include"
    export CGO_LDFLAGS="-L$SRC/internal/compute -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64"
    go build -tags cuda -o "$WORK/bin/modelbench-cuda" ./cmd/modelbench
  fi
  FAK_CUDA_TF32=1 LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    "$WORK/bin/modelbench-cuda" -gguf "$MODEL" -backend cuda -require-non-reference \
      -prefill-sizes 512 -prefill-reps 5 -decode-steps 128 -decode-reps 5 -decode-prompt 16 \
      -out "$WORK/fak-cuda-tf32-report.json"
  python3 "$WORK/norm_fak.py" fak-cuda-tf32 "$WORK/fak-cuda-tf32-report.json" "$WORK/engine-fak-cuda-tf32.json"
}''',
}
_ENGINE_FN = {"llama": "engine_llama", "vllm": "engine_vllm",
              "fak-cpu": "engine_fak_cpu", "fak-cuda": "engine_fak_cuda",
              "fak-cuda-q8": "engine_fak_cuda_q8", "fak-cuda-tf32": "engine_fak_cuda_tf32"}


_DRIVER_PREAMBLE = r'''#!/usr/bin/env bash
# fak GCP multi-engine bench driver (rendered by gcp_bench.py; run as: sudo bash).
# set -u + pipefail, but NOT -e globally: a failed engine must not abort the others.
set -uo pipefail
WORK=/opt/gcp-bench
SRC="$WORK/fak"
mkdir -p "$WORK/bin"
log(){ echo "[vm-bench $(date -u +%H:%M:%S)] $*"; }
# Build the error JSON with python3 (present on the DLVM base image pre-apt) so a
# message with a quote/backslash can't emit invalid JSON; fall back to a static blob.
fatal(){ python3 -c 'import json,sys;open(sys.argv[2],"w").write(json.dumps({"schema":"fak.gcp-vm-bench.v2","error":sys.argv[1]}))' "$1" "$WORK/result.json" 2>/dev/null || echo '{"schema":"fak.gcp-vm-bench.v2","error":"fatal (see serial log)"}' > "$WORK/result.json"; log "FATAL: $1"; exit 1; }

# CUDA_CC is the GPU compute-capability number (89=L4, 100=B200) -- NOT the C compiler
# (build_cuda.sh reads $CC for that), so it is deliberately named to never collide.
CUDA_CC="@@CC@@"
MODEL_FILE="@@HF_FILE@@"
MODEL_REPO="@@HF_REPO@@"

log "== fak GCP multi-engine bench (arch sm_$CUDA_CC) =="
nvidia-smi -L || fatal "no GPU / driver"
GPUS=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
log "GPU: $GPUS"

export DEBIAN_FRONTEND=noninteractive
# The DLVM image runs unattended-upgrades at boot; a bare apt-get races it and dies
# on the dpkg lock. Wait for every apt/dpkg lock to clear, then retry.
wait_apt(){ for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1 || return 0; log "waiting for apt lock ($i)"; sleep 10; done; return 0; }
apt_do(){ wait_apt; for i in 1 2 3 4 5; do apt-get "$@" && return 0; log "apt retry $i"; sleep 15; wait_apt; done; return 1; }
log "apt deps"
apt_do update -y -q || fatal "apt update failed"
apt_do install -y -q build-essential cmake git libcurl4-openssl-dev python3-pip jq || fatal "apt install failed"

# Go toolchain (the DLVM has none). Install the latest release; GOTOOLCHAIN=auto
# then fetches the exact version go.mod asks for if it is newer.
export PATH="/usr/local/cuda/bin:/usr/local/go/bin:$PATH"
export GOTOOLCHAIN=auto
if ! command -v go >/dev/null 2>&1; then
  log "installing Go"
  GOVER=$(curl -fsSL 'https://go.dev/VERSION?m=text' | head -1) || true
  [ -n "$GOVER" ] || fatal "go version lookup failed (empty)"
  curl -fsSL "https://go.dev/dl/${GOVER}.linux-amd64.tar.gz" -o /tmp/go.tgz || fatal "go download failed ($GOVER)"
  rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
fi
go version || fatal "no go toolchain"

# fak source: the SCP'd tarball of the local working tree (repo is private; this is
# the exact code under test). Extracts to $WORK/fak.
log "extracting fak source"
[ -f /tmp/fak-src.tgz ] || fatal "fak source tarball /tmp/fak-src.tgz missing (scp step?)"
rm -rf "$SRC"; tar -C "$WORK" -xzf /tmp/fak-src.tgz
[ -f "$SRC/go.mod" ] || fatal "extracted source has no fak/go.mod"

# write the small python helpers
cat > "$WORK/norm_llama.py" <<'NORMLLAMA'
@@PY_NORM_LLAMA@@
NORMLLAMA
cat > "$WORK/norm_fak.py" <<'NORMFAK'
@@PY_NORM_FAK@@
NORMFAK
cat > "$WORK/norm_vllm.py" <<'NORMVLLM'
@@PY_NORM_VLLM@@
NORMVLLM
cat > "$WORK/diag.py" <<'DIAGPY'
@@PY_DIAG@@
DIAGPY
cat > "$WORK/combine.py" <<'COMBINEPY'
@@PY_COMBINE@@
COMBINEPY

# shared model fetch (every engine runs the SAME model on the SAME GPU)
log "fetching model @@HF_REPO@@/@@HF_FILE@@"
pip install -q -U huggingface_hub >/dev/null 2>&1 || pip3 install -q -U huggingface_hub >/dev/null 2>&1 || true
MODEL=$(python3 - <<'HFPY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="@@HF_REPO@@", filename="@@HF_FILE@@"))
HFPY
)
[ -n "$MODEL" ] && [ -f "$MODEL" ] || fatal "model fetch failed"
log "model at $MODEL"

# run one engine in an isolated subshell (set -e local to it) so a failure is
# captured and diagnosed, never fatal to the rest.
run_one(){
  local key="$1" fn="$2"
  log "== engine $key: start =="
  ( set -e; "$fn" ) > "$WORK/$key.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    log "== engine $key: OK =="
  else
    log "== engine $key: FAILED rc=$rc (diagnosing) =="
    python3 "$WORK/diag.py" "$key" "$WORK/$key.log" "$WORK/engine-$key.json" || true
  fi
}

'''

_DRIVER_TAIL = r'''
# ---- combine every engine row into one result.json -------------------------
python3 "$WORK/combine.py" "$WORK" "$GPUS" "$MODEL_FILE" "sm_$CUDA_CC"
log "== done; result at $WORK/result.json =="
cat "$WORK/result.json"
'''


def render_driver_script(engine_keys: list[str], hf_repo: str, hf_file: str, cc: str) -> str:
    """Assemble the shared preamble + selected engine fragments + the combiner."""
    body = _DRIVER_PREAMBLE
    body = body.replace("@@CC@@", cc)
    body = body.replace("@@HF_REPO@@", hf_repo)
    body = body.replace("@@HF_FILE@@", hf_file)
    body = body.replace("@@PY_NORM_LLAMA@@", _PY_NORM_LLAMA.strip("\n"))
    body = body.replace("@@PY_NORM_FAK@@", _PY_NORM_FAK.strip("\n"))
    body = body.replace("@@PY_NORM_VLLM@@", _PY_NORM_VLLM.strip("\n"))
    body = body.replace("@@PY_DIAG@@", _PY_DIAG.strip("\n"))
    body = body.replace("@@PY_COMBINE@@", _PY_COMBINE.strip("\n"))

    defs = "\n\n".join(_ENGINE_BASH[k] for k in engine_keys)
    runs = "\n".join(f'run_one {k} {_ENGINE_FN[k]}' for k in engine_keys)
    return body + defs + "\n\n" + runs + "\n" + _DRIVER_TAIL


def make_source_tarball(dest: Path, dry_run: bool) -> Path:
    """Tar the local fak/ tree (the exact code under test) for SCP to the VM.

    Excludes the heavy, regenerable, or VM-fetched bits: .git, .cache (venvs,
    cached models), build artifacts, and any bundled model weights -- the VM
    fetches its own GGUF. Uses Python's tarfile (no shell, fully portable); a dir
    excluded by the filter is not walked, so the multi-GB .cache is never stat'd.
    """
    fak_dir = FAK_SRC
    # `experiments/` is benchmark DATA (handoff tarballs, GGUFs, oracle dumps --
    # hundreds of MB); the VM builds cmd/modelbench from SOURCE and fetches its own
    # model, so none of it is needed. Excluding it is what keeps the tarball small.
    exclude_dir_names = {".git", ".cache", "node_modules", "__pycache__", "experiments"}
    exclude_suffixes = (".test", ".exe", ".o", ".a", ".gguf", ".safetensors",
                        ".bin", ".pt", ".pth", ".onnx", ".log",
                        ".tgz", ".tar", ".gz", ".zip", ".so", ".dylib", ".dll", ".wasm")

    def keep(ti: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        parts = ti.name.split("/")
        if any(p in exclude_dir_names for p in parts):
            return None
        # When the source IS the repo root, a compiled `fak` binary may sit at the
        # root; it has no excluded suffix, so drop it explicitly (it would tar as
        # fak/fak and bloat the tarball / shadow nothing useful on the VM).
        if FAK_SRC == ROOT and ti.name == "fak/fak":
            return None
        if ti.isfile() and ti.name.endswith(exclude_suffixes):
            return None
        return ti

    if dry_run:
        log(f"DRY-RUN: would tar {fak_dir} -> {dest} (excluding {sorted(exclude_dir_names)})")
        return dest
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(fak_dir, arcname="fak", filter=keep)
    size_mb = dest.stat().st_size / 1e6
    log(f"source tarball {dest.name}: {size_mb:.1f} MB")
    if size_mb > 200:
        log(f"WARNING: source tarball is {size_mb:.0f} MB -- check excludes (a stray model/venv?)")
    return dest


def instance_name(tier: gcp_accel.AccelTier) -> str:
    return f"fak-bench-{tier.slug}-{stamp()}".lower().replace("_", "-")


def machine_id_for(tier: gcp_accel.AccelTier) -> str:
    return f"gcp-{tier.slug}"


def _zone_name(zone: str) -> str:
    return (zone or "").rstrip("/").split("/")[-1]


def _delete_instance_command(runner: Runner, name: str, zone: str) -> str:
    cmd = runner._base() + [
        "compute", "instances", "delete", name, f"--zone={zone}", "--quiet",
    ]
    return " ".join(shlex.quote(c) for c in cmd)


def warn_preexisting_bench_instances(runner: Runner) -> None:
    """Surface fak-bench-* VMs leaked by prior launchers before creating a new one."""
    proc = runner.run(
        [
            "compute", "instances", "list",
            "--filter=name~^fak-bench-",
            "--format=json(name,zone,status,creationTimestamp)",
        ],
        capture=True, timeout=120, check=False,
    )
    if runner.dry_run:
        return
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        suffix = f": {msg}" if msg else ""
        log(f"WARNING: could not check for pre-existing fak-bench-* VMs{suffix}")
        return
    out = (proc.stdout or "").strip()
    if not out:
        return
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        log("WARNING: could not parse pre-existing fak-bench-* VM list")
        return
    if not isinstance(rows, list):
        log("WARNING: unexpected pre-existing fak-bench-* VM list shape")
        return
    leaked = [r for r in rows if str(r.get("name", "")).startswith("fak-bench-")]
    if not leaked:
        return
    log(f"WARNING: found {len(leaked)} pre-existing fak-bench-* VM(s) before launch:")
    for row in leaked:
        name = str(row.get("name") or "")
        zone = _zone_name(str(row.get("zone") or ""))
        status = row.get("status") or "UNKNOWN"
        created = row.get("creationTimestamp") or "unknown-created"
        log(f"  {name} zone={zone} status={status} created={created}")
        if zone:
            log(f"    delete: {_delete_instance_command(runner, name, zone)}")
        else:
            log("    delete: re-run `gcloud compute instances list` to resolve its zone")


def resolve_instance_ip(runner: Runner, name: str, zone: str) -> Optional[str]:
    """Return the current external NAT IP for an existing bench VM."""
    proc = runner.run(
        [
            "compute", "instances", "describe", name,
            f"--zone={zone}",
            "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
        ],
        capture=True, timeout=120, check=False,
    )
    if runner.dry_run:
        return None
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        suffix = f": {msg}" if msg else ""
        log(f"STOP: could not resolve current IP for {name} in {zone}{suffix}")
        return None
    ip = (proc.stdout or "").strip()
    if not ip:
        log(f"STOP: {name} in {zone} has no external NAT IP")
        return None
    return ip.splitlines()[0].strip()


def resolve_tier(args, runner: Runner) -> Optional[gcp_accel.AccelTier]:
    """Pick the tier to launch: explicit pin, proof, or probe the ladder."""
    if args.proof:
        t = gcp_accel.proof_tier()
        log(f"--proof: cheapest tier {t.slug} ({t.gpu_label})")
        return t
    if args.tier:
        t = gcp_accel.by_slug(args.tier)
        if not t:
            slugs = ", ".join(x.slug for x in gcp_accel.TIERS)
            log(f"unknown --tier {args.tier!r}; valid: {slugs}")
            return None
        log(f"--tier pinned: {t.slug} ({t.gpu_label})")
        return t

    # No pin: probe and take the recommended (best provisionable) tier.
    log("probing the fallback ladder for a provisionable tier...")
    rep = probe.probe(project=runner.project, account=runner.account,
                      all_tiers=not args.blackwell, zone_override=args.zone)
    if rep.get("stale_auth"):
        log("STOP: GCP credentials are stale. Fix once, interactively:")
        log("    gcloud auth login")
        return None
    rec = rep.get("recommended")
    if not rec:
        if args.blackwell:
            log("STOP: no Blackwell tier (B200/GB200) is provisionable in this "
                "project today. Request quota/reservation, or drop --blackwell to "
                "allow the H200/H100/L4 fallback.")
        else:
            log("STOP: no tier in the ladder has live GPU quota. Request quota "
                "(IAM > Quotas) or a reservation; or use --proof to try L4.")
        return None
    t = gcp_accel.by_slug(rec)
    log(f"probe recommends: {t.slug} ({t.gpu_label})")
    return t


def provision(runner: Runner, tier: gcp_accel.AccelTier, name: str, zone: str,
              startup_path: Path, spot: bool, max_run_hours: float = 2.0) -> None:
    image_family, image_project = gcp_accel.boot_image()
    args = [
        "compute", "instances", "create", name,
        f"--zone={zone}",
        f"--machine-type={tier.machine_type}",
        "--maintenance-policy=TERMINATE",
        f"--image-family={image_family}",
        f"--image-project={image_project}",
        "--boot-disk-size=200GB",
        "--boot-disk-type=pd-ssd",
        f"--metadata-from-file=startup-script={startup_path}",
        "--metadata=install-nvidia-driver=True",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
    ]
    # Accelerator-optimized machine types (a4/a4x/a3/g2) carry their GPUs
    # IMPLICITLY in the machine type -- gcloud REJECTS --accelerator on them
    # ("Accelerators are not supported for machine type a4-highgpu-8g"). Only the
    # older general-purpose families (N1 + a T4/V100/etc.) take an explicit
    # --accelerator. Pass it iff the machine type isn't self-accelerated.
    if not tier.machine_type.startswith(("a4", "a3-", "a2-", "g2-")):
        args.append(f"--accelerator={gcp_accel.accelerator_flag(tier)}")
    # A4/A4X/A3U require local SSD; attach a couple for scratch + model cache.
    if tier.machine_type.startswith(("a4", "a3-ultra", "a3-high")):
        args += ["--local-ssd=interface=NVME", "--local-ssd=interface=NVME"]
    if spot:
        args.append("--provisioning-model=SPOT")
    # GCP-side max lifetime: the dominant leak mode is the LAUNCHER process being
    # killed before its Python `finally` teardown runs (a `finally` does NOT run on
    # SIGKILL/external terminate) -- a leaked L4 once billed ~10h this way. A
    # server-side --max-run-duration is the belt-and-suspenders that survives a dead
    # launcher: GCP itself deletes the instance after the TTL no matter what the
    # client does. A healthy run finishes well inside it (the SSH driver run is capped
    # at 5400s/90min), so the TTL only ever bites a stranded VM. DELETE (not STOP) is
    # correct: a STOPped GPU VM still bills for its boot disk. --max-run-duration
    # composes with --maintenance-policy=TERMINATE (that governs host-maintenance
    # events; this governs lifetime). --instance-termination-action is REQUIRED by
    # --max-run-duration and must appear exactly once even alongside --spot.
    if max_run_hours and max_run_hours > 0:
        args.append(f"--max-run-duration={int(round(max_run_hours * 3600))}s")
    if (max_run_hours and max_run_hours > 0) or spot:
        args.append("--instance-termination-action=DELETE")
    runner.run(args, timeout=900)


def teardown(runner: Runner, name: str, zone: str) -> None:
    """Delete the instance. Idempotent: a missing instance is success."""
    try:
        runner.run(
            ["compute", "instances", "delete", name, f"--zone={zone}", "--quiet"],
            capture=True, timeout=600, check=True,
        )
        log(f"torn down: {name}")
    except RuntimeError as e:
        if "was not found" in str(e) or "notFound" in str(e):
            log(f"teardown: {name} already gone")
        else:
            log(f"WARNING: teardown of {name} failed -- delete it by hand:")
            log(f"    gcloud compute instances delete {name} --zone={zone} --quiet")
            raise


def wait_for_ssh(runner: Runner, name: str, zone: str, timeout_s: int = 300) -> bool:
    """Poll `gcloud compute ssh -- true` until it connects or times out."""
    if runner.dry_run:
        log("DRY-RUN: skip SSH wait")
        return True
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        # NOTE: no "--ssh-flag=-o ..." here. On Windows gcloud shells out to
        # plink, which rejects OpenSSH-style "-o ConnectTimeout=..." ("unknown
        # option -o"). The per-attempt subprocess timeout below bounds the wait
        # portably instead.
        proc = runner.run(
            ["compute", "ssh", name, f"--zone={zone}", "--command=true",
             "--tunnel-through-iap"],
            capture=True, timeout=60, check=False,
        )
        if proc.returncode == 0:
            log(f"SSH up after {attempt} attempt(s)")
            return True
        time.sleep(10)
    log("SSH never came up within timeout")
    return False


def run_driver_over_ssh(runner: Runner, name: str, zone: str,
                        tarball: Path, driver: Path) -> Optional[dict]:
    """SCP the source + driver to the VM, run the driver, read result.json back."""
    if runner.dry_run:
        log("DRY-RUN: would scp source+driver, run `sudo bash /tmp/bench-driver.sh`, read result.json")
        return {"schema": "fak.gcp-vm-bench.v2", "_dry_run": True}
    log("shipping source + driver to the VM")
    runner.scp(str(tarball), name, "/tmp/fak-src.tgz", zone)
    runner.scp(str(driver), name, "/tmp/bench-driver.sh", zone)
    # The build+bench of three engines (incl. two CUDA builds) is the slow part;
    # give it a long ceiling. Logs stream; we tail the tail.
    runner.run(
        ["compute", "ssh", name, f"--zone={zone}", "--tunnel-through-iap",
         "--command=sudo bash /tmp/bench-driver.sh 2>&1 | tail -n 80"],
        timeout=5400, check=False,
    )
    # Read result.json back, retrying on a transient SSH/IAP flake. The benchmark
    # just burned expensive GPU minutes and teardown follows this read, so a single
    # stalled `sudo cat` must NOT be allowed to discard a real result: the read is
    # cheap and idempotent, so retry it a few times before giving up. We accept the
    # parse only when the JSON carries the expected schema key -- that distinguishes a
    # complete file from a partial/empty read (disk full mid-write, fatal() fallback).
    last = ""
    for attempt in range(3):
        proc = runner.run(
            ["compute", "ssh", name, f"--zone={zone}", "--tunnel-through-iap",
             "--command=sudo cat /opt/gcp-bench/result.json"],
            capture=True, timeout=120, check=False,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out:
            try:
                parsed = json.loads(out)
            except json.JSONDecodeError:
                last = "result.json was not valid JSON"
            else:
                if isinstance(parsed, dict) and parsed.get("schema"):
                    return parsed
                last = "result.json missing 'schema' key (partial/empty write?)"
        else:
            last = "could not read result.json from VM"
        if attempt < 2:
            log(f"{last} -- retrying readback ({attempt + 2}/3)")
            time.sleep(8)
    log(last)
    return None


def collect(tier: gcp_accel.AccelTier, zone: str, result: dict, dry_run: bool,
            engine_keys: list[str]) -> Path:
    """Fold the VM result into the benchmark catalog layout + register the machine."""
    machine_id = machine_id_for(tier)
    run_id = f"{machine_id}-{tier.accelerator_type}-{stamp()}"
    run_dir = RUNS_DIR / machine_id / f"{stamp()}-gcp"
    tags = ["gcp", "gpu", tier.arch] + (["blackwell"] if tier.blackwell else [])
    tags += [f"engine-{k}" for k in engine_keys]
    manifest = {
        "$schema": "benchmark/run-manifest.v1",
        "run_id": run_id,
        "machine_id": machine_id,
        "timestamp": stamp(),
        "git": {"rev": "unknown", "branch": "master", "dirty": False},
        "harness": {"name": "fak-gcp-bench", "version": "2"},
        "model": {"name": result.get("model", DEFAULT_HF_FILE), "precision": "Q8_0"},
        "config": {"zone": zone, "machine_type": tier.machine_type,
                   "accelerator": gcp_accel.accelerator_flag(tier),
                   "engines": engine_keys},
        "tags": tags,
        "artifacts": {"gpu": "result.json"},
    }
    specs = {
        "$schema": "benchmark/machine-specs.v1",
        "machine_id": machine_id,
        "hostname": f"gcp-{tier.machine_type}",
        "registered_at": utc_now(),
        "hardware": {
            "gpu": {"model": tier.gpu_label, "count": tier.gpu_count,
                    "memory_gb": tier.gpu_mem_gb_each,
                    "compute_capability": tier.compute_capability},
            "cpu": {"architecture": "x86_64", "cores_logical": tier.vcpus},
            "ram_gb": tier.host_mem_gb,
        },
        "os": {"name": "GCP Deep Learning VM (CUDA)"},
        "cloud": {"provider": "gcp", "machine_type": tier.machine_type, "zone": zone},
        "tags": [tier.arch, "cuda", "gcp"] + (["blackwell"] if tier.blackwell else []),
    }
    if dry_run:
        log(f"DRY-RUN: would write {run_dir}/manifest.json + result.json + specs")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    spec_dir = MACHINES_DIR / machine_id
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    log(f"collected -> {run_dir}")
    # A compact head-to-head line so the operator sees the story without opening JSON.
    engines = result.get("engines") or {}
    for k in engine_keys:
        e = engines.get(k) or {}
        if e.get("ok"):
            log(f"  {k:9} prefill={e.get('prefill_tok_per_sec')} tok/s  decode={e.get('decode_tok_per_sec')} tok/s")
        else:
            log(f"  {k:9} FAILED: {e.get('error', 'no row')}")
    log("fold into catalog with: python tools/bench_catalog.py update")
    return run_dir


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default=None, help="pin a tier slug (else probe the ladder)")
    ap.add_argument("--blackwell", action="store_true",
                    help="strict: only B200/GB200; STOP if neither is provisionable")
    ap.add_argument("--proof", action="store_true",
                    help="use the cheapest (L4) tier to prove the pipeline")
    ap.add_argument("--engine", default="all",
                    help="which engine(s): llama, vllm, fak-cpu, fak-cuda, fak-cuda-q8, "
                         "fak-cuda-tf32, fak, all, or a comma list (default: all). vLLM, "
                         "fak-cuda-q8 and fak-cuda-tf32 are opt-in -- NOT in 'all'; run the "
                         "apples-to-apples Q8 head-to-head with --engine llama,fak-cuda,fak-cuda-q8, "
                         "or the TF32 prefill lever with --engine llama,fak-cuda,fak-cuda-tf32")
    ap.add_argument("--zone", default=None, help="override zone (else tier default)")
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT") or None)
    ap.add_argument("--account", default=os.environ.get("GCP_ACCOUNT") or None)
    ap.add_argument("--resolve-ip", metavar="INSTANCE",
                    help="print the current external IP for an existing GCP bench VM "
                         "and exit; intended for bench_nodes.json resolve_cmd")
    ap.add_argument("--spot", action="store_true",
                    help="request a Spot VM (cheaper, preemptible)")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--hf-file", default=DEFAULT_HF_FILE)
    ap.add_argument("--fak-ref", default="master")
    ap.add_argument("--keep", action="store_true",
                    help="do NOT tear down the VM (debug). Prints the delete command.")
    ap.add_argument("--max-run-hours", type=float, default=2.0,
                    help="GCP-side auto-delete TTL (hours) so a killed launcher can't "
                         "leak the GPU VM; 0 disables (default: 2.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every gcloud command; create/run/delete nothing")
    args = ap.parse_args(argv)

    if args.resolve_ip:
        if not args.zone:
            log("STOP: --resolve-ip requires --zone")
            return 2
        runner = Runner(args.dry_run, args.project, args.account)
        with redirect_stdout(sys.stderr):
            ip = resolve_instance_ip(runner, args.resolve_ip, args.zone)
        if not ip:
            return 1
        print(ip)
        return 0

    try:
        engine_keys = resolve_engines(args.engine)
    except ValueError as e:
        log(f"STOP: {e}")
        return 2
    if not engine_keys:
        log("STOP: no engines selected")
        return 2

    runner = Runner(args.dry_run, args.project, args.account)
    warn_preexisting_bench_instances(runner)

    tier = resolve_tier(args, runner)
    if not tier:
        return 2

    # A CPU-only engine set doesn't need a GPU, but every tier here HAS one; the
    # needs_cuda flag just records intent + lets a future CPU-only tier skip the
    # CUDA engines. On a real GPU tier all engines run.
    zone = args.zone or tier.common_zones[0]
    name = instance_name(tier)
    cc = tier.compute_capability
    log(f"plan: {tier.slug} ({tier.gpu_count}x {tier.gpu_label}) in {zone}, "
        f"instance {name}, engines={engine_keys}, ~${tier.approx_usd_per_hour:.0f}/hr")

    startup_body = render_setup_script()
    driver_body = render_driver_script(engine_keys, args.hf_repo, args.hf_file, cc)
    startup_path = Path(ROOT / "tools" / f".gcp-startup-{stamp()}.sh")
    driver_path = Path(ROOT / "tools" / f".gcp-driver-{stamp()}.sh")
    tarball_path = Path(ROOT / "tools" / f".gcp-src-{stamp()}.tgz")

    result: Optional[dict] = None
    provisioned = False
    try:
        # Local setup is inside the try so the finally cleans its temp files even if
        # tarball creation raises.
        if not args.dry_run:
            # MUST be LF: a CRLF script breaks bash on the Linux VM (see write_lf).
            write_lf(startup_path, startup_body)
            write_lf(driver_path, driver_body)
        make_source_tarball(tarball_path, args.dry_run)
        # Fail-fast BEFORE any billable create if the source root has no go.mod: the
        # on-VM driver guards on `$SRC/go.mod` and would fatal AFTER the VM is booting
        # and billing. Turn that silent-spend into a free local error.
        if not (FAK_SRC / "go.mod").is_file():
            log(f"aborting: bench source root {FAK_SRC} has no go.mod (nothing to build)")
            return 2
        # Arm teardown around the ENTIRE window the VM could exist: set the flag BEFORE
        # the create call, so a partial create that then raises (or a Ctrl-C mid-create)
        # still hits the idempotent teardown in finally. Deleting a never-created
        # instance is a no-op, so arming early is always safe.
        provisioned = True
        provision(runner, tier, name, zone, startup_path, args.spot, args.max_run_hours)
        if not wait_for_ssh(runner, name, zone):
            log("aborting: SSH never came up")
            return 1
        result = run_driver_over_ssh(runner, name, zone, tarball_path, driver_path)
        if result is None and not args.dry_run:
            log("bench produced no result")
            return 1
        collect(tier, zone, result or {}, args.dry_run, engine_keys)
    except KeyboardInterrupt:
        log("interrupted -- tearing down before exit")
        return 130
    finally:
        if not args.dry_run:
            for p in (startup_path, driver_path, tarball_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        if provisioned and not args.keep:
            teardown(runner, name, zone)
        elif provisioned and args.keep:
            ttl = (f" (GCP auto-deletes in ~{args.max_run_hours:g}h)"
                   if args.max_run_hours and args.max_run_hours > 0 else "")
            log(f"--keep: VM {name} LEFT RUNNING and billing{ttl}. Delete sooner with:")
            log(f"    gcloud compute instances delete {name} --zone={zone} --quiet")

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
