# Provenance-locked multi-arm benchmark runner

`fak armbench` runs every comparison arm from one immutable JSON manifest. It is
the shared harness for the Caveman/Ponytail benchmark work under #6674; later
issues add provider/judge adapters, statistics, and a ledger without inventing
per-comparison scripts. `fak armbench import-fixtures` is the immutable-input
boundary that feeds the runner.

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

The corpus hash is `sha256:` plus SHA-256 over the compact JSON encoding of the
ordered `tasks` array. The runner recomputes it before the first provider call.
Unknown JSON fields, trailing JSON values, missing/invalid pins, an edited
corpus, bundled or unnamed capabilities, duplicate resume keys, missing raw
request/response/judge evidence, and incomparable manifests all fail closed.

Each raw trial row retains request, response, provider-reported input/output
usage and cost, wall/TTFT/inter-token latency (with availability bits), cache
counters, failure/retry data, and raw judge evidence. Each arm also records its
one-time setup cost; the human and JSON reports show its per-trial amortization.

## Immutable Caveman/Ponytail fixture import

The importer has no floating revision, repository, or path flag. One command
fetches only the reviewed declarations below, verifies the raw response bytes,
and materializes an ordinary `manifest.json` + `corpus.json` pair:

```text
fak armbench import-fixtures --suite all --review-license "JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4=MIT"
```

Use `--suite caveman` or `--suite ponytail` to materialize one input. `--json`
emits `fak.armbench.fixture-import/1`; result paths are relative to `--store` so
the report is portable. Without `--store`, bytes go under the operating
system's user-cache directory at `fak/armbench-fixtures`, never under repository
scratch. A supplied store is refused if its resolved path (including symlinks)
lands anywhere inside the repository.

The layout is content-addressed:

```text
STORE/
  blobs/sha256/<retrieved-content-hash>
  inputs/sha256/<manifest+corpus-content-hash>/
    manifest.json
    corpus.json
    sources/<owner>/<repo>/<commit>/<declared-path>
```

The command's human output prints the exact downstream command. Its shape is
the runner contract established by #6676:

```text
fak armbench run --manifest STORE/inputs/sha256/ID/manifest.json --corpus STORE/inputs/sha256/ID/corpus.json --out STORE/inputs/sha256/ID/run.json
```

Every emitted source row records its canonical URL, repository, full commit,
declared path, retrieved-content hash, detected license boundary and boundary
hash, license-review status, retrieval date, byte-normalization statement, and
local fixture path (`local_path`). The retrieval date is the UTC calendar date.
Normalization is deliberately
`none; byte-exact upstream response`.

### License review boundary

Ponytail's pinned root `LICENSE` asserts MIT, so its review status is
`repository-asserted`. Caveman's repository-level license detector is
`NOASSERTION` and the pinned tree has a mixed boundary: `LICENSE` plus
`LICENSING.md` assign the imported benchmark/skill/documentation/test surfaces
to MIT while runtime surfaces are BSL. Therefore every Caveman import requires
the exact revision-bound review token shown above. The importer never admits
Caveman runtime paths under `engine/`, `proxy/`, `cacheengine/`, `rewriter/`,
`browse/`, `mcp/`, `shrink/`, `mem/`, or `shared/platform/`.

The fetched boundary itself is checked, not merely described:

- Caveman `LICENSE` + `LICENSING.md` boundary:
  `sha256:695deeb180b5a1e28a4eafd822d6a86b8673a642eee5e2865e1a3cc6cf43d3df`.
- Ponytail `LICENSE` boundary:
  `sha256:df8847f0cfdbc2f8d3b5a322e9fc8c6f3f411729c00e531d267fa5f78da51ae1`.

For a reproducible composite boundary digest, the importer hashes each listed
file in order as an unsigned 64-bit little-endian byte length followed by the
raw bytes. The single-file Ponytail boundary uses the same framing.

### Exact pinned declarations

All hashes below are SHA-256 over the byte-exact upstream response.

**Caveman — `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`**

| Role | Path | SHA-256 |
|---|---|---|
| harness | `benchmarks/run.py` | `530a387918418713e64ded97794f41a1ffe6a01e833a69d2cb447bf4640facce` |
| corpus | `benchmarks/prompts.json` | `773e557f9187363c44e7e5aae2d27268720bcd8772865e119825078b06da93d7` |
| active system prompt | `skills/caveman/SKILL.md` | `daf9cec496ebd039809d8236f99f17fa1b4beaadf8ce4e2d532d0da51d70afce` |
| evaluation metadata | `benchmarks/requirements.txt` | `994d50c7e2e135d7621b812c929bb6efca7d8f1ddd1d41476caa71ec3f1eecd1` |
| evaluation metadata | `docs/HONEST-NUMBERS.md` | `740ac5e8bb6722c7a5d45f82d8308c5fa6ced93a06eb97040f712363882f7c59` |
| evaluation metadata | `evals/README.md` | `253bde57a74c83f57d0a5f53ec5ed77cb2cbae1c23a364999f44259c444ab376` |
| evaluation contract | `tests/test_benchmark_contract.py` | `27a37ae418f00555761ccdb085078933750c87bd9a4c44f93209f29e0b18c678` |
| license | `LICENSE` | `f0abc56b6f49ab2e285bb6e6723f028abb7ebd4fe0e242bbdc2b4dded0ace8b9` |
| license map | `LICENSING.md` | `d4804b40d29ec31ee03b163e68eec134e1967c7d2cc53d8068ea5c3fabbbf7b4` |

**Ponytail — `DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3`**

| Role | Path | SHA-256 |
|---|---|---|
| Promptfoo config | `benchmarks/promptfooconfig.yaml` | `fc292f68c5727f306f5ba1ce74b161631e416ce71b66a67d9241a59556c3e00d` |
| Promptfoo config | `benchmarks/promptfooconfig.gpt.yaml` | `1e196e71efc3c77c54efb10d58634a9e1d1fe58c7e8755ee712825e1c6b940fb` |
| Promptfoo config | `benchmarks/promptfooconfig.gpt-newest.yaml` | `e3bcfbf34eb148d81bfd3c985d88ea7e7f71c9989b74b9588c8136488ed18f72` |
| Promptfoo config | `benchmarks/promptfooconfig.gemini.yaml` | `2c1578b259fc3efb755d1c1534f2a01ebe75ea05e6fc2cd6701ea68e99affce6` |
| corpus | `benchmarks/prompts.json` | `6137aae013b057d38c90b38af17e6b22d57011b85d0be5d549045df196fa37ac` |
| baseline arm | `benchmarks/arms/baseline.js` | `ef0f81f670425705ab3195609947aa64890890cb078d7669780afe2228da8740` |
| Caveman arm | `benchmarks/arms/caveman.js` | `b3793a7549c9217e9e9b89b2ed5e94813a9c71297386c77a177d4e2ef3e13d1a` |
| Ponytail arm | `benchmarks/arms/ponytail.js` | `9091a2e794ad1c5c7c4c9b68a3bb4aa995fcce6d0f09aad37a2febdd95640b82` |
| Caveman system prompt | `benchmarks/arms/caveman-SKILL.md` | `09ebdef35a85d058f8eba04c6a3d91079ac8dabc3d45d265694f128c808f7648` |
| Ponytail system prompt | `skills/ponytail/SKILL.md` | `1316a2f3f95741d2300b116fe0c2d81ce4a9568656ed0a62643f54aaf09957f2` |
| line-count definition | `benchmarks/loc.js` | `c3c346f836eec0ba759a100a5c1166be2362fcf37017470f3c9f9b872ab44234` |
| behavior definition | `benchmarks/behavior.js` | `9364957a4fade600cd423098e115e2187d12ebcaefc54900f4b368819f33efd2` |
| behavior config | `benchmarks/behavior.yaml` | `7f0edb875c56872fa14707a5e096a0f951dfc0b5f6de82fb7bfe505b9fd45c4f` |
| correctness definition | `benchmarks/correctness.js` | `7a17d8c904eab1813a220279c31ef51598214735923d178bed9beb92e9cf0230` |
| robustness definition | `benchmarks/robustness-audit.js` | `e9a7e9eb6b087e60e2fed07e61370fbd1bfca76b16bb95c86170cffedcebff96` |
| agentic tasks | `benchmarks/agentic/tasks.py` | `68f473557695f69a036cd6fdb5d8e9ec51aef407183416545b29823c1e3190a2` |
| agentic judge | `benchmarks/agentic/judge.py` | `c845548290c062dac5bef93ba3231e26b35be62a07fdebabec61833c4a20b6c0` |
| license | `LICENSE` | `fb1bc6909ac3ef82d5c22106e32ef682b0cff66788fa915fb9b53b15c9d2f3ab` |

### Fail-closed behavior and proof

The importer returns exit `3` with a closed reason token for:

- changed upstream bytes (`FIXTURE_HASH_MISMATCH`);
- a declared path that moved or vanished (`FIXTURE_PATH_MOVED`);
- missing/inconsistent license or normalization metadata
  (`FIXTURE_LICENSE_METADATA_MISSING`);
- omitted Caveman review (`FIXTURE_LICENSE_REVIEW_REQUIRED`);
- a requested store inside the repository (`FIXTURE_STORE_INSIDE_REPO`);
- an admitted Caveman runtime path (`FIXTURE_RESTRICTED_PATH`); or
- any edited, missing, or extra byte/path already under a content address
  (`FIXTURE_LOCAL_MUTATION`).

Tests use only an in-process local fixture server; CI never fetches GitHub. The
captured live command report at
[`docs/_witnesses/armbench-import-2026-08-14.json`](_witnesses/armbench-import-2026-08-14.json)
materialized both exact revisions. The importer tests then load its emitted
manifest/corpus through the normal armbench decoder and execute them with the
network-free fake provider.

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
