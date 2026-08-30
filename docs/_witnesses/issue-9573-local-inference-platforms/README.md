# Issue 9573: dated local-inference platform inventory

This witness is the smallest machine-readable and rendered inventory of representative
64/128 GB local-inference platforms needed to seed the already-landed
[`LongContextEstimatorInput`](../../../internal/modelperfobs/long_context_estimator.go).
AMD Strix Halo is first, followed by an Apple unified-memory system, an NVIDIA compact
appliance, and a CPU/high-memory workstation.

The authority is [`inventory.json`](inventory.json). The compact operator view is
[`LOCAL-INFERENCE-PLATFORMS-2026-08-27.md`](../../notes/LOCAL-INFERENCE-PLATFORMS-2026-08-27.md).

## What the inventory does

- Preserves a source URL, publisher/title, source event date where known, observation/access
  date (`2026-08-27`), source state, platform context, refresh trigger, and field-borrow
  license disposition.
- Types decision data as `measured`, `official_spec_derived`, `analytically_derived`, or
  `assumption_speculative`.
- Keeps vendor-labeled GB, normalized estimator bytes, and GiB ranges distinct.
- Separates installed shared/unified memory from discrete VRAM and from memory actually
  available to one inference process.
- Records official peak bandwidth separately from sustainable measurements. A missing
  workload-matched measurement is `null`, not an invented utilization percentage.
- Records date-specific list/street prices, including configuration omissions and preorder
  status.

## Compact inventory

| Platform | Memory architecture | Marketed memory | Conservative usable range | Advertised bandwidth | Sustainable evidence | Power boundary | Observed price |
|---|---|---:|---:|---:|---|---|---:|
| Framework Desktop Ryzen AI Max+ 395 | 256-bit LPDDR5x-8000 shared CPU/iGPU memory | 64 GB | 48–52 GiB | 256 GB/s | ~215 GB/s maximum GPU MBW measured on the related pre-production 128 GB system | 120 W sustained / 140 W boost processor | $1,959 DIY system selection |
| Framework Desktop Ryzen AI Max+ 395 | 256-bit LPDDR5x-8000 shared CPU/iGPU memory | 128 GB | 92–104 GiB | 256 GB/s | ~215 GB/s maximum GPU MBW on pre-production Framework hardware | 120 W sustained / 140 W boost processor | $3,449 DIY system selection |
| Apple Mac Studio M5 Max | Apple unified CPU/GPU memory | 128 GB | 96–108 GiB | 614 GB/s | `null` — preorder, not shipping until 2026-09-22 | 480 W system maximum continuous | $5,399 configured estimate |
| `NVIDIA DGX Spark` | coherent LPDDR5x shared by Grace CPU and Blackwell GPU | 128 GB | 104–112 GiB | 273 GB/s | `null` for GPU inference; CPU STREAM context retained separately | 140 W GB10 TDP / 240 W supply | $4,699 reported list; $5,299.99 observed retail |
| Dell Precision 7875 | eight-channel CPU-attached DDR5-5200 ECC RDIMM | 128 GB | 108–116 GiB | 332.8 GB/s theoretical | `null` on the Dell; 206.1 GB/s comparable-topology context only | 350 W CPU / 1,350 W configured PSU | $19,897.33, no storage |

Prices are nominal USD before tax. Framework rows omit storage, OS, fan, power cable, tiles,
and optional ports. The Dell row explicitly has no storage. The Apple price is analytically
reconstructed from current base and configure-to-order deltas because the dynamic 128 GB page
did not expose a stable total in the text fetch.

## Estimator handoff

The inventory maps directly only where the evidence is safe:

- `UsableMemoryBytes` ← one bound from
  `memory.conservative_usable_inference_memory_bytes`, after replacing the planning reserve
  with an observed reserve for the actual OS/backend.
- `BandwidthBytesPerSec` ← a workload-matched measured range. Do **not** insert the advertised
  peak as both bounds. The inventory intentionally leaves `estimator_ready_range` null.
- `ComputeFLOPS` ← precision- and kernel-matched measured or defensible peak range. NPU TOPS,
  sparse FP4 PFLOPS, and GPU FP16 TFLOPS are not interchangeable.
- `Efficiency` remains workload/backend-specific; this inventory does not hide it inside a
  hardware number.

These are hardware inputs, **not measured fak-native Qwen3.8 or GLM-5.3 performance**.

## Provenance boundary

The source ledger has 12 authoritative entries and 7 third-party entries. Official product
specifications and price/configuration pages establish product facts and list boundaries.
Third-party rows are used only for measurements, a current retail observation, or a price
delta that the dynamic official storefront did not render stably.

The field-borrow inward check found the estimator itself **PRESENT** at
[`internal/modelperfobs/long_context_estimator.go`](../../../internal/modelperfobs/long_context_estimator.go)
and a dated purchase-class platform inventory **ABSENT** from the existing
[`Hardware Catalog`](../../notes/HARDWARE-CATALOG.md),
[`Hardware Matrix`](../../HARDWARE-MATRIX.md), and
[`experiments/benchmark/machines`](../../../experiments/benchmark/machines/) records. Durable
study search returned no matching receipt. The dev self-index was unavailable because the
separate `fak-dev` executable was not installed; raw repository inspection supplied the
fallback witness.

All external sources are factual inspiration only (`INSPIRE-ONLY`); no vendor or
third-party expressive code or prose was copied.

## Validation

From the repository root:

```bash
python3 -m json.tool docs/_witnesses/issue-9573-local-inference-platforms/inventory.json >/dev/null

python3 - <<'PY'
import json
from pathlib import Path

p = Path("docs/_witnesses/issue-9573-local-inference-platforms/inventory.json")
d = json.loads(p.read_text())
source_ids = {s["id"] for s in d["sources"]}
assert len(source_ids) == len(d["sources"])
assert sum(s["authority"] == "authoritative" for s in d["sources"]) == 12
assert sum(s["authority"] == "third_party" for s in d["sources"]) == 7
assert d["platforms"][0]["category"] == "amd_strix_halo"
assert d["platforms"][1]["category"] == "amd_strix_halo"
assert {p["memory"]["marketed_physical_memory"]["value"] for p in d["platforms"][:2]} == {64, 128}
for platform in d["platforms"]:
    assert platform["memory"]["conservative_usable_inference_memory_bytes"]["min"] > 0
    assert platform["memory"]["conservative_usable_inference_memory_bytes"]["min"] <= platform["memory"]["conservative_usable_inference_memory_bytes"]["max"]
    text = json.dumps(platform)
    def walk(v):
        if isinstance(v, dict):
            if "source_ids" in v:
                assert set(v["source_ids"]) <= source_ids
            for child in v.values():
                walk(child)
        elif isinstance(v, list):
            for child in v:
                walk(child)
    walk(platform)
print(f"{len(d['platforms'])} platforms; {len(d['sources'])} sources; references OK")
PY
```

External HTTP status is only a reachability witness, never proof that a claim is true. The final
parallel probe returned HTTP 200 for 16 of 19 source URLs. Framework's configurator and Dell's
storefront returned HTTP 403 to the probe after both had rendered during collection; the Best Buy
listing timed out. NVIDIA Marketplace also timed out in a separate price probe, so the `NVIDIA DGX Spark` list
price is explicitly sourced to the dated third-party report rather than falsely marked as a
direct official observation.

## Limitations

1. Shared/unified memory is not VRAM. Installed capacity, GPU-addressable policy, and safe
   process working set are different quantities.
2. OS/reserve ranges are conservative planning assumptions. Replace them with observed
   committed/resident memory and pressure before a fit claim.
3. The Strix Halo ~215 GB/s measurement is a maximum-bandwidth probe, not a completed-job
   inference range. CPU STREAM results for Halo and Spark are not GPU decode bandwidth.
4. M5 Max was announced two days before the as-of date and was still a preorder, so sustainable
   measurements and street pricing are intentionally null.
5. Software maturity is a dated qualitative classification, not a promise that every model,
   quantization, or kernel works.
6. **TOPS is not a throughput prediction.** Precision, sparsity, accumulation, batch, model
   architecture, memory traffic, and software support all change the result.
7. API-versus-local comparison must ultimately use quality-qualified completed-job cost and
   completion time, including setup, queueing, retries, rejected work, failures, power, and
   operator burden—not peak token rate or purchase price alone.
