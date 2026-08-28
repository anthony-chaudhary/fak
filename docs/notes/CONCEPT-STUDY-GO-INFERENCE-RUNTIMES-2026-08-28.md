---
title: "Study: Go inference runtimes on GitHub"
description: "A delegated, source-pinned study of 23 Go inference repositories, centered on the boundary between Go orchestration, Go execution, dynamic native bindings, and translated provenance."
---

# Study: Go inference runtimes on GitHub

**Outcome:** issue [#9846](https://github.com/anthony-chaudhary/fak/issues/9846) ships a
human-readable and machine-readable inventory of 23 repositories. The decisive finding is that
the phrase “Go inference runtime” hides four materially different systems: native Go execution,
Go governance over native workers, binding/framework layers, and research implementations.
Durable receipt: `study_154039c4f551c649ee58f5dea721dae58c7a08488490ca0bc40915a313f9ce7c`.

Inventory:

- [`go-inference-runtimes-github-2026-08-28.md`](../research/inventory/go-inference-runtimes-github-2026-08-28.md)
- [`go-inference-runtimes-github-2026-08-28.json`](../research/inventory/go-inference-runtimes-github-2026-08-28.json)

Companions: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) ·
[`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) ·
[#9846](https://github.com/anthony-chaudhary/fak/issues/9846)

## Frame

For: fak maintainers choosing Go-native mechanisms, comparison references, and interoperability
targets.

Problem: GitHub results and project descriptions routinely conflate Go APIs, no-CGo dynamic
bindings, Go-authored model graphs, generated Go, and native Go tensor execution.

Today: repeat ad hoc searches or consult adjacent landscape notes that mention Ollama and LocalAI
without a pinned Go-runtime corpus.

Better because: every row identifies what Go owns, where model math executes, exact revision,
license boundary, formats/hardware, activity posture, and code/test anchors.

Witness: both inventory files parse/read cleanly; each of the 23 rows has a full SHA and immutable
implementation/test URL; the issue and durable study receipt are read back after creation.

Centrality: **Enabling** — native-inference and decode-parity decisions need an honest prior-art
and interop map.

- P1 Context: preserved; this documentation-only pass reduces repeated discovery.
- P2 Net value: advanced by preventing invalid comparisons; it makes no speed or quality claim.
- P3 Adaptation: advanced through dated pins, refresh triggers, and a machine-readable schema.
- P4 Operations: advanced through indexed, auditable artifacts and a durable receipt.

## Acquisition and coverage

Three delegated readers used distinct evidence lanes:

1. native Go engines and translated/static execution;
2. Go-governed native backends and binding frameworks;
3. general frameworks, research implementations, search coverage, and completeness criticism.

Together they acquired and pinned more than 30 plausible repositories in allocated scratch,
then reduced them to 23 included rows plus explicit exclusions. The coordinator independently
read GitHub repository/commit metadata for every material row and re-read boundary-defining code
for the major families before folding worker claims. The installed DOS doctor could not emit its
workspace JSON because the installed binary rejected the repository's newer `dos.toml` stamp key;
fresh GitHub API and immutable-content read-back supplied the non-agent witness instead.

Source classes checked, observed 2026-08-28:

- README/docs for the map, followed by implementation and tests for claims;
- default-branch revision, releases/tags, recent history, and material open/closed issue/PR state;
- root and file-level licenses, submodules/gitlinks, vendored/generated/transpiled provenance;
- Go ownership versus the actual CPU/GPU/native-library execution boundary;
- model formats, representative architectures, hardware routes, and maintenance posture;
- a dated GitHub search ledger and family-level completeness critic.

No README-only runtime row survived. Search-result counts are discovery evidence, not a quality
score. Scratch clones were reaped by the delegated readers.

## On-axis candidate dispositions

| Candidate | Source fact and immutable anchor | Axis | Their-worldview reason | fak witness on-axis | Disposition | Filed |
|---|---|---|---|---|---|---|
| Execution-boundary-labeled Go runtime inventory | Yzma's loader dynamically opens native libraries even though its Go build is no-CGo ([`loader.go:12-18@c4fed78`](https://github.com/hybridgroup/yzma/blob/c4fed7865c4c5cb116d7ae9105a1765bc0398803/pkg/loader/loader.go#L12-L18)); GoMLX stores the selected backend in its executor ([`exec.go:126-135@f847c57`](https://github.com/gomlx/gomlx/blob/f847c57c9a4e10c1c66c737bbc44625fd4d1e538/core/graph/exec.go#L126-L135)). | Provenance-honest discovery: distinguish control language, execution language, backend, and implementation origin. | Go users optimize deployment friction, portability, ecosystem reuse, or kernel ownership differently; one language label cannot preserve those choices. | **ABSENT** before this pass: `fak capabilities` returned general routing only; docs/leaves/verbs/claims indexes and raw grep found adjacent notes but no dedicated inventory. Seam `docs/research/inventory/`. | **DEFAULT** documentation spine; implemented here. | [#9846](https://github.com/anthony-chaudhary/fak/issues/9846) |
| Per-route engine identity in inference evidence | Hugot can execute a model through native ORT, native XLA, or GoMLX's Go backend ([`model_gomlx.go:42-58@e085914`](https://github.com/knights-analytics/hugot/blob/e08591494e5383c534e1c3cef92ac9d766e461b0/backends/model_gomlx.go#L42-L58)). | A result must name the selected engine/backend, not only the Go package used to request inference. | Backend interchangeability is valuable only when support and performance are evaluated per route. | **PRESENT-on-axis**: `internal/agent/chat.go:460` defines `NativeInferenceReceipt`; `internal/abi/registry.go:1021` exposes registered engine identities; native receipt tests bind measured output to the engine path. | **DEFAULT**, already shipped; no duplicate issue. | — |
| No-CGo dynamic FFI as a deployment technique | ONNX Runtime Go uses CGo `dlopen` ([`setup_env.go:62-75@8ce7fa5`](https://github.com/yalue/onnxruntime_go/blob/8ce7fa5882bb3ac038bacdd53bfe02da3691899f/setup_env.go#L62-L75)); Yzma and Onnxer replace CGo with dynamic FFI but keep foreign computation. | Build/deployment friction, not native execution ownership. | These projects prioritize cross-compilation and simpler distribution while retaining mature kernels. | **DIVERGENT** for fak-native work: fak's invariant requires fak to retain execution, kernel, memory, scheduling and cache ownership. It remains valid as an explicit interoperability/optional adapter technique, never a silent fallback. | **OPTIONAL-MODULE / WATCH** only when an explicit integration names the foreign engine. | — |
| Translated llama.cpp as static Go execution | goccy/go-llama's invocation table calls generated `wasm2go` functions ([`engine.go:67-80@6b608ae`](https://github.com/goccy/go-llama/blob/6b608ae6947a4f6a27b2f05f85cb0c56ac3253b9/internal/engine.go#L67-L80)). | Static single-binary deployment versus authorship and native kernel ownership. | It serves users who value Go packaging and no runtime dependency more than preserving a Go-designed engine. | **DIVERGENT**: it is a valuable parity/reference mechanism but adopting it as fak-native would yield implementation ownership to translated llama.cpp and inherit wasm32 memory constraints. | **WATCH** as reference; do not file a native fallback. | — |
| Portable native Go execution exists beyond demos | Born has a Go CPU backend ([`backend.go:11-25@b1237ec`](https://github.com/born-ml/born/blob/b1237ec5135d1108f56af6dfd5068638068ad8d3/internal/backend/cpu/backend.go#L11-L25)); goinfer owns its layer loop ([`model.go:560-575@bf6bdc3`](https://github.com/townsendmerino/goinfer/blob/bf6bdc3afc633ac8c118a7a982fc00c4ff8d2bdc/decoder/model.go#L560-L575)); rembed owns its encoder forward ([`bert.go:649-665@19d673b`](https://github.com/rostamlabs/rembed/blob/19d673b357bd8c244aa1a6d17d83769224fa98c9/internal/model/bert.go#L649-L665)). | Whether native Go is only pedagogical or can be a practical runtime substrate. | Their users want static binaries, transparent execution and low integration burden, accepting narrower maturity or model/hardware coverage. | **PRESENT-on-axis**: fak already owns its in-kernel Go forward path and produces strict native receipts. These repositories broaden prior art; they do not establish a missing capability or performance lead. | **WATCH** for mechanisms and matched benchmarks; no duplicate feature issue. | — |

The single ABSENT candidate was the inventory itself, and it is the spine tracked by #9846. The
other findings are PRESENT-on-axis or deliberate divergences under fak's explicit native-inference
ownership rule, so no additional implementation issue is justified by this pass.

## Worldview findings

- Go developers use “single binary” to mean at least three different things: no external process,
  no CGo at build time, or no foreign execution engine. The projects show those are independent
  properties.
- General frameworks optimize replaceable backends and model authoring; model-specific runtimes
  optimize artifact compatibility and serving; bindings optimize ecosystem reuse. A global winner
  would erase the user/job distinction.
- Small, fast-moving native engines can be technically substantial while having almost no adoption
  or independent operating-envelope evidence. Activity and code presence justify `WATCH`, not a
  performance conclusion.

## Completeness critic

The final corpus covers Go-governed servers; native Go model engines; native Go tensor/autograd
frameworks; ONNX Runtime CGo and `purego` styles; llama.cpp CGo, `purego`, and translated-to-Go
styles; libtorch; a format importer with pluggable execution; and representative Llama-2/Llama-3
research implementations.

The omitted candidates were duplicate provenance families, downstream applications, companion
modules, narrow fixed-model consumers, or unlicensed experiments. GitHub search can still miss Go
subtrees in non-Go-primary monorepos or projects whose descriptions omit the searched terms. That
is the refresh condition: repeat the query ledger and re-run the family critic when a material new
runtime, release, or execution boundary appears.

## Honest limits

- No project was benchmarked. Source performance claims remain unverified.
- Model and hardware coverage are exact-revision code/readme conclusions, not live conformance
  runs over every combination.
- License dispositions are a good-faith technical classification, not legal advice.
- The machine inventory is the row-level authority; this note records the method and fak-facing
  decisions.
