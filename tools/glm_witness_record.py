#!/usr/bin/env python3
"""glm_witness_record.py — turn a GLM-5.2 GPU-witness run log into a structured,
content-hashed record (schema glm-gpu-witness/1, see tools/schemas/glm-gpu-witness.v1.json).

The witness runner (tools/dgx_glm_gpu_witness.sh) emits only human text: timestamped
`say()` banners and raw `go test -tags cuda -v` stdout (the cosine/argmax lines) plus a
final `GLM GPU WITNESS DONE head=.. rc1=.. rc2=.. rc3=.. -> rc=..` line. That is a SELF-REPORT
an operator hand-transcribes into prose. This parser converts it ONCE into a machine-readable
artifact a tool can validate, diff run-to-run, and assert on — and stamps a SHA256 over the
exact raw log so the record is provably the one parsed from that log (and so a bridge fetch can
detect a truncated/split transfer).

Honesty: each per-test `cosine` is the output of a `go test` gate that FAILS unless the cosine
clears its floor (class=approx) — so the per-test rc, and the overall rc, are kernel-enforced,
not author-claimed. `dos verify` binds the COMMIT carrying this record (git ancestry + stamp
grammar); it does not open the JSON or re-check the number. The number's correctness is
re-checkable only by re-running the test.

Pure stdlib, no wall-clock read (pass --utc), no network. Determinism: identical (log, --utc,
flags) -> identical JSON.

Usage:
  python tools/glm_witness_record.py /tmp/fakglm/run.log --utc 2026-06-24T13:53:08Z
  python tools/glm_witness_record.py run.log --utc <iso> --public        # a100 rollup, no node
  python tools/glm_witness_record.py --self-test                          # offline golden check
"""
import argparse
import hashlib
import json
import re
import sys

RECORD_TOOL = "glm_witness_record.py@1"
SCHEMA = "glm-gpu-witness/1"

# --- line grammars (matched against the witness runner's stdout) -----------------------------
RE_NODE = re.compile(r"node (?P<node>\S+) gpu=(?P<gpu>\d+) arch=(?P<arch>\S+)")
RE_HEAD = re.compile(r"=== \[[^\]]*\] HEAD (?P<sha>[0-9a-f]{7,40})")
RE_BUILT = re.compile(r"built (?P<bytes>\d+) byte libfakcuda\.a")
RE_BUILD_OK = re.compile(r"\[cuda\] OK build")
# banner: "go test -tags cuda -run <TestName> (<human description>) ==="
RE_BANNER = re.compile(r"-run (?P<name>\S+) \((?P<desc>[^)]*)\)")
RE_RUN = re.compile(r"=== RUN\s+(?P<name>\S+)")
# cosine line: "... cosine=1.000000 argmax cpu=40 cuda=40 tier=sm_80 class=approx"
RE_COSINE = re.compile(r"cosine=(?P<cos>[0-9.]+)")
RE_ARGMAX = re.compile(r"argmax cpu=(?P<cpu>\d+) (?:cuda|hybrid|device)=(?P<dev>\d+)")
RE_TIER = re.compile(r"tier=(?P<tier>\S+)")
RE_CLASS = re.compile(r"class=(?P<cls>\S+)")
RE_VERDICT = re.compile(r"--- (?P<v>PASS|FAIL): (?P<name>\S+)")
RE_SUMMARY = re.compile(
    r"GLM GPU WITNESS DONE head=(?P<head>\S+)\s+"
    r"rc1=(?P<rc1>\d+) rc2=(?P<rc2>\d+) rc3=(?P<rc3>\d+) -> rc=(?P<rc>\d+)"
)


def _to_int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_log(log_text):
    """Parse witness-runner stdout into the structured record fields (no utc / machine_id / sha;
    those are stamped by build_record). Returns a dict."""
    node = arch = head = None
    gpu_count = None
    build = {}
    # ordered per-test accumulation, keyed by test_name
    tests = {}
    order = []
    cur = None  # current test_name (set by banner or RUN line)

    for raw in log_text.splitlines():
        line = raw.rstrip("\n")

        m = RE_NODE.search(line)
        if m and node is None:
            node = m.group("node")
            arch = m.group("arch")

        m = RE_HEAD.search(line)
        if m and head is None:
            head = m.group("sha")

        m = RE_BUILT.search(line)
        if m:
            build["libfakcuda_bytes"] = int(m.group("bytes"))
        if RE_BUILD_OK.search(line):
            build["ok"] = True

        m = RE_BANNER.search(line)
        if m:
            cur = m.group("name")
            t = tests.setdefault(cur, {"test_name": cur})
            if cur not in order:
                order.append(cur)
            # keep the first/banner description as path_exercised
            t.setdefault("path_exercised", m.group("desc").strip())
            continue

        m = RE_RUN.search(line)
        if m:
            cur = m.group("name")
            tests.setdefault(cur, {"test_name": cur})
            if cur not in order:
                order.append(cur)
            continue

        if cur is not None and "cosine=" in line:
            t = tests[cur]
            mc = RE_COSINE.search(line)
            if mc:
                t["cosine"] = float(mc.group("cos"))
            ma = RE_ARGMAX.search(line)
            if ma:
                t["argmax_cpu"] = int(ma.group("cpu"))
                t["argmax_device"] = int(ma.group("dev"))
            mt = RE_TIER.search(line)
            if mt:
                t["tier"] = mt.group("tier")
            mcl = RE_CLASS.search(line)
            if mcl:
                t["class"] = mcl.group("cls")

        m = RE_VERDICT.search(line)
        if m:
            name = m.group("name")
            t = tests.setdefault(name, {"test_name": name})
            if name not in order:
                order.append(name)
            t["verdict"] = m.group("v")
            t["rc"] = 0 if m.group("v") == "PASS" else 1

    summary = {}
    for line in log_text.splitlines():
        m = RE_SUMMARY.search(line)
        if m:
            summary = {
                "head": m.group("head"),
                "rc1": int(m.group("rc1")),
                "rc2": int(m.group("rc2")),
                "rc3": int(m.group("rc3")),
                "rc": int(m.group("rc")),
            }
    if summary.get("head") and head is None:
        head = summary["head"]

    # finalize per-test list (fill defaults the schema expects)
    test_list = []
    for name in order:
        t = tests[name]
        t.setdefault("cosine", None)
        t.setdefault("gate", None)
        t.setdefault("argmax_cpu", None)
        t.setdefault("argmax_device", None)
        t.setdefault("set_equality", None)
        if "verdict" not in t:
            # no PASS/FAIL line seen -> unknown; treat as FAIL (fail-loud)
            t["verdict"] = "FAIL"
            t["rc"] = 1
        test_list.append(t)

    return {
        "node": node,
        "arch": arch,
        "head_sha": head,
        "build": build or None,
        "gpu_count": gpu_count,
        "tests": test_list,
        "summary": summary,
    }


def build_record(log_bytes, utc, machine_id, model_name, precision, public, head_subject=None,
                 environment=None):
    """Assemble the full glm-gpu-witness/1 record from raw log bytes + caller-stamped fields."""
    log_text = log_bytes.decode("utf-8", "replace")
    parsed = parse_log(log_text)
    content_sha = hashlib.sha256(log_bytes).hexdigest()

    tests = parsed["tests"]
    summary = parsed["summary"]
    # overall rc: prefer the runner summary; else 0 iff all tests PASS
    if summary:
        rc = summary["rc"]
    else:
        rc = 0 if tests and all(t.get("rc", 1) == 0 for t in tests) else 1
    verdict = "PASS" if rc == 0 else "FAIL"

    rec = {
        "schema": SCHEMA,
        "utc": utc,
        "machine_id": machine_id,
        "head_sha": parsed["head_sha"],
        "arch": parsed["arch"],
        "model": {"name": model_name, "precision": precision},
        "tests": tests,
        "rc": rc,
        "verdict": verdict,
        "content_sha256": content_sha,
        "log_bytes": len(log_bytes),
        "record_tool": RECORD_TOOL,
    }
    if head_subject:
        rec["head_subject"] = head_subject
    if parsed["build"]:
        rec["build"] = parsed["build"]
    if environment:
        rec["environment"] = environment
    # node: private only. Public rollup drops it.
    if not public and parsed["node"]:
        rec["node"] = parsed["node"]
    return rec


# --- offline golden sample (witness output shape; hostname genericized — real node names stay
# in the private record, never in public source) -- HEAD 7889a5b, sm_80, 2026-06-24 ------------
GOLDEN_LOG = """=== [13:53:08] node gpu-node.local gpu=0 arch=sm_80 ===
=== [13:53:08] clone origin/main ===
Cloning into '/tmp/fakglm3/src'...
=== [13:53:11] HEAD 7889a5b ===
=== [13:53:11] build libfakcuda.a (sm_80) ===
[cuda] nvcc compile kernels (sm_80) ...
[cuda] built 217836 byte libfakcuda.a
[cuda] go build -tags cuda ./internal/compute/ ...
[cuda] OK build
-rw-r--r-- 1 root root 217836 Jun 24 06:53 internal/compute/libfakcuda.a
=== [13:53:16] go test -tags cuda -run TestCUDAGLMMoeDsaBackendForward (all-device GLM-5.2 DSA forward) ===
=== RUN   TestCUDAGLMMoeDsaBackendForward
    glm_dsa_cuda_test.go:63: GLM-MoE-DSA forward with MoE/FFN+head + DSA attention projections (k_q8_gemm) + DSA sparse attention (k_dsa_sparse_attend) on cuda backend: cosine=1.000000 argmax cpu=40 cuda=40 tier=sm_80 class=approx
--- PASS: TestCUDAGLMMoeDsaBackendForward (0.16s)
PASS
ok  \tgithub.com/anthony-chaudhary/fak/internal/model\t0.972s
=== [13:53:23] go test -tags cuda -run TestCUDAGLMDsaCPUOffloadHybrid (cpu-offload hybrid) ===
=== RUN   TestCUDAGLMDsaCPUOffloadHybrid
    glm_dsa_offload_cuda_test.go:98: GLM-DSA --n-cpu-moe hybrid on cuda: experts host-offloaded (OFF the GPU); router+MLA projections+DSA sparse attention on k_q8_gemm/k_dsa_sparse_attend; cosine=1.000000 argmax cpu=40 hybrid=40 tier=sm_80 class=approx
--- PASS: TestCUDAGLMDsaCPUOffloadHybrid (0.28s)
PASS
ok  \tgithub.com/anthony-chaudhary/fak/internal/model\t0.949s
=== [13:53:25] go test -tags cuda -run TestCUDAGLMMoeDsaIndexSelectMatches (device index score+topk) ===
=== RUN   TestCUDAGLMMoeDsaIndexSelectMatches
    glm_dsa_cuda_test.go:122: GLM-MoE-DSA decode with index SELECTION on cuda backend (k_dsa_index_score + k_dsa_index_topk, f64-accumulated): cosine=1.000000 argmax cpu=40 cuda=40 tier=sm_80 class=approx
--- PASS: TestCUDAGLMMoeDsaIndexSelectMatches (0.13s)
PASS
ok  \tgithub.com/anthony-chaudhary/fak/internal/model\t0.843s
=== [13:53:27] GLM GPU WITNESS DONE head=7889a5b rc1=0 rc2=0 rc3=0 -> rc=0 ===
""".replace("\\t", "\t")


def self_test():
    rec = build_record(
        GOLDEN_LOG.encode("utf-8"),
        utc="2026-06-24T13:53:08Z",
        machine_id="dgx",
        model_name="glm_moe_dsa",
        precision="q8",
        public=False,
    )
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(rec["schema"] == SCHEMA, "schema")
    check(rec["head_sha"] == "7889a5b", f"head_sha={rec.get('head_sha')}")
    check(rec["arch"] == "sm_80", f"arch={rec.get('arch')}")
    check(rec["node"] == "gpu-node.local", f"node={rec.get('node')}")
    check(rec["rc"] == 0 and rec["verdict"] == "PASS", "overall rc/verdict")
    check(len(rec["tests"]) == 3, f"n_tests={len(rec['tests'])}")
    names = [t["test_name"] for t in rec["tests"]]
    check(names == [
        "TestCUDAGLMMoeDsaBackendForward",
        "TestCUDAGLMDsaCPUOffloadHybrid",
        "TestCUDAGLMMoeDsaIndexSelectMatches",
    ], f"names={names}")
    for t in rec["tests"]:
        check(t["cosine"] == 1.0, f"{t['test_name']} cosine={t.get('cosine')}")
        check(t["verdict"] == "PASS", f"{t['test_name']} verdict")
        check(t["argmax_cpu"] == 40 and t["argmax_device"] == 40,
              f"{t['test_name']} argmax {t.get('argmax_cpu')}/{t.get('argmax_device')}")
        check(t.get("class") == "approx", f"{t['test_name']} class")
    check(rec["build"]["libfakcuda_bytes"] == 217836, "build bytes")
    check(rec["build"]["ok"] is True, "build ok")
    check(re.fullmatch(r"[a-f0-9]{64}", rec["content_sha256"]) is not None, "sha256 shape")
    # public rollup drops node + flips machine_id
    pub = build_record(GOLDEN_LOG.encode("utf-8"), utc="2026-06-24T13:53:08Z",
                       machine_id="a100", model_name="glm_moe_dsa", precision="q8", public=True)
    check("node" not in pub, "public drops node")
    check(pub["machine_id"] == "a100", "public machine_id")
    check(pub["content_sha256"] == rec["content_sha256"], "sha stable across public flag")

    if fails:
        print("SELF-TEST FAIL:")
        for f in fails:
            print("  -", f)
        print(json.dumps(rec, indent=2))
        return 1
    print("SELF-TEST OK — glm-gpu-witness/1 parsed: 3 tests, all cosine=1.0 PASS, rc=0, "
          f"head=7889a5b, sha={rec['content_sha256'][:12]}…")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Emit a glm-gpu-witness/1 record from a witness log.")
    ap.add_argument("log", nargs="?", help="path to the witness run.log ('-' for stdin)")
    ap.add_argument("--utc", help="ISO-8601 UTC run stamp (REQUIRED for a real record; pass "
                    "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ) from the caller)")
    ap.add_argument("--machine-id", default="dgx", help="catalog machine id (default dgx; "
                    "use a100 for the public rollup)")
    ap.add_argument("--model", default="glm_moe_dsa")
    ap.add_argument("--precision", default="q8", choices=["f32", "f16", "bf16", "q8", "q4", "int8"])
    ap.add_argument("--public", action="store_true", help="public rollup: drop node, default "
                    "machine_id to a100")
    ap.add_argument("-o", "--out", help="write JSON here (default stdout)")
    ap.add_argument("--self-test", action="store_true", help="run the offline golden check + exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.log:
        ap.error("log path required (or --self-test)")
    if not args.utc:
        ap.error("--utc is required (determinism: the parser never reads the wall clock)")
    machine_id = args.machine_id
    if args.public and machine_id == "dgx":
        machine_id = "a100"

    data = sys.stdin.buffer.read() if args.log == "-" else open(args.log, "rb").read()
    rec = build_record(data, utc=args.utc, machine_id=machine_id, model_name=args.model,
                       precision=args.precision, public=args.public)
    out = json.dumps(rec, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"wrote {args.out} ({rec['verdict']} rc={rec['rc']} "
              f"{len(rec['tests'])} tests sha={rec['content_sha256'][:12]}…)")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
