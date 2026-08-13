# Provenance-locked multi-arm benchmark runner

`fak armbench` runs every comparison arm from one immutable JSON manifest. It is
the shared harness for the Caveman/Ponytail benchmark work under #6674; later
issues add fixture importers, provider/judge adapters, statistics, and a ledger
without inventing per-comparison scripts.

## Exact consumer command

Every later benchmark issue supplies a manifest and its content-addressed corpus
to this command shape (paths are illustrative; the flags are exact):

```text
fak armbench run --manifest INPUT/manifest.json --corpus INPUT/corpus.json --out ARTIFACTS/run.json
```

Resume the same manifest without re-running completed trial keys:

```text
fak armbench run --manifest INPUT/manifest.json --corpus INPUT/corpus.json --out ARTIFACTS/run.json --resume ARTIFACTS/prior-run.json
```

The provider is selected by `manifest.model.provider`. `--provider NAME` is an
optional assertion and is refused if it disagrees with the manifest. The first
spine registers only `fake`; later provider adapters extend that registry while
keeping this command unchanged.

Roll up an already captured raw ledger, or check whether two complete manifests
describe the same experiment:

```text
fak armbench report --run ARTIFACTS/run.json
fak armbench report --run ARTIFACTS/run.json --json
fak armbench compare --a ARTIFACTS/run-a.json --b ARTIFACTS/run-b.json --json
```

`compare` refuses any execution/provenance drift and names every changed field.
Arms are compared inside a run; changing the arm contract creates a different
manifest identity.

## Manifest contract

The strict `fak.armbench.manifest/1` schema records:

- upstream repository, full commit SHA, path, and retrieved-content SHA-256;
- provider, model snapshot, region, temperature, top-p, sampling seed, and max
  output tokens;
- ordered-corpus SHA-256 and task count, plus judge identifier/SHA-256;
- trial count, paired-order strategy (`counterbalanced` or seeded `randomized`),
  seed, and concurrency;
- OS, architecture, host class, fak version, and pricing date; and
- one or more typed arms: `baseline`, `upstream_treatment`, `fak_passthrough`,
  or `fak_capability`. A capability arm names exactly one capability; other arm
  kinds name none.

The deterministic spine binds the issue's pinned revisions (metadata only; the
fixture importer in #6677 owns materialization and license review):

| Source | Revision and path | Retrieved-content SHA-256 |
|---|---|---|
| Caveman | `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4:benchmarks/run.py` | `530a387918418713e64ded97794f41a1ffe6a01e833a69d2cb447bf4640facce` |
| Ponytail | `DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3:benchmarks/arms/baseline.js` | `ef0f81f670425705ab3195609947aa64890890cb078d7669780afe2228da8740` |

The corpus hash is `sha256:` plus SHA-256 over the compact JSON encoding of the
ordered `tasks` array. The runner recomputes it before the first provider call.
Unknown JSON fields, trailing JSON values, missing/invalid pins, an edited
corpus, bundled or unnamed capabilities, duplicate resume keys, missing raw
request/response/judge evidence, and incomparable manifests all fail closed.

Each raw trial row retains request, response, provider-reported input/output
usage and cost, wall/TTFT/inter-token latency (with availability bits), cache
counters, failure/retry data, and raw judge evidence. Each arm also records its
one-time setup cost; the human and JSON reports show its per-trial amortization.

## Deterministic fake-provider proof

Create a runnable demo without a key, network, model, or GPU:

```text
fak armbench emit-demo --dir _scratch/armbench-demo
fak armbench run --manifest _scratch/armbench-demo/manifest.json --corpus _scratch/armbench-demo/corpus.json --out _scratch/armbench-demo/run.json
fak armbench selfcheck --json
```

The selfcheck executes baseline, the pinned Caveman treatment, fak passthrough,
and one `ctxmmu-paging` arm. It captures the complete raw ledger and report,
proves model/prompt/judge/corpus/capability mutations move the identity, exercises
resume and paired ordering, and witnesses the raw-evidence and capability
refusals. Its byte-for-byte deterministic committed capture is
[`docs/_witnesses/armbench-selfcheck-2026-08-13.json`](_witnesses/armbench-selfcheck-2026-08-13.json).

Exit codes: `0` success, `2` usage, `3` a typed fail-closed refusal, and `1` an
I/O or internal runtime error.
