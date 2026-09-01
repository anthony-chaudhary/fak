# OpenCode Go Ox Alpha coding dogfood — 2026-08-22

## Verdict

**Useful as an optional, zero-token-price coding route; not a default yet.** The live model correctly reviewed the provider/account wiring, but its reasoning overhead and latency were high for a very small task, and two low output caps produced no visible answer before a larger cap completed.

## Value frame

- **For:** fak operators with an OpenCode Go subscription who want to spend that allowance from fak or another OpenAI-compatible harness.
- **Problem:** the subscription was absent from fak's built-in provider-account roster, and the temporary free Ox Alpha model had no logical route.
- **Today:** add an env-reference-only account and an `ox-alpha-free` binding.
- **Better because:** fak can resolve the subscription independently of the OpenCode CLI; native OpenCode can still address it as `opencode-go/ox-alpha-free`.
- **Witness:** `TestDefaultRosterIsValidAndMixesProviders` plus the live calls below.

## Provenance

Observed 2026-08-22 from OpenCode's provider documentation and repository commit `anomalyco/opencode@ba72a6ff2b62`:

- Display name: **Ox Alpha Free**; availability: **limited time**.
- Native OpenCode model id: `opencode-go/ox-alpha-free`.
- OpenAI-compatible model id: `ox-alpha-free`.
- Chat endpoint: `https://opencode.ai/zen/go/v1/chat/completions`.

The credential remains outside the repository. fak stores only the environment-variable name `OPENCODE_GO_API_KEY`.

## Live results

Direct, OpenAI-compatible requests used the operator-provided disposable key only in process memory:

| Task | Result | Latency | Usage |
|---|---|---:|---:|
| Exact-response connectivity check | `OX_READY` | 2.5 s | 94 prompt, 32 completion, 10 reasoning, 64 cached |
| Review fak's proposed account + binding | `PASS`; no blocking defect | 80.0 s | 167 prompt, 2,275 completion, 764 reasoning, 64 cached |

Two earlier review attempts capped at 300 and 1,200 completion tokens ended with `finish_reason=length` and **zero visible content**; the successful request needed a 4,000-token cap. That is meaningful dogfood evidence: budget enough hidden reasoning, and do not make Ox Alpha the default for latency-sensitive small edits until broader trials show better efficiency.

The completed review correctly checked endpoint composition, secret indirection, and exact model binding. It suggested confirming the env-var convention and rate/context metadata; the env name is fak's own non-secret handle, while OpenCode currently documents Ox Alpha limits as unspecified. No unproven limits are encoded.

## Recommendation

Classify this integration **OPTIONAL-MODULE**:

- Use `ox-alpha-free` when free allowance matters more than latency.
- Keep stronger established coding models as the default route.
- Re-evaluate before the limited-time offer ends or when OpenCode publishes stable limits/model identity.

## Medium-scale issue-review and gardening audit

A later native OpenCode session tested Ox Alpha on a materially larger task: review the whole repository, identify the largest issues, and file the findings. The local OpenCode export is session `ses_fd6053091ffeuxugge92jO0pl2` (created 2026-08-22 07:58 PDT, completed 08:32 PDT). The transcript itself remains in OpenCode's local store because it contains working-tree and GitHub interaction detail; this note records the independently reproducible audit.

### Trajectory shape

| Measure | Observed |
|---|---:|
| Wall time | 34m 10s |
| Model | `opencode-go/ox-alpha-free` |
| Provider cost reported by OpenCode | $0 |
| Input / output / reasoning tokens | 129,898 / 11,251 / 1,418 |
| Cache-read tokens | 2,343,872 |
| Messages / tool calls / tool errors | 43 / 51 / 2 |
| Mutating draft operations (`write` + `edit`) | 17 |

The run produced a concise final synthesis and then filed five issues (#8588-#8592). That demonstrates useful sustained tool use and enough repository comprehension to find one real structural theme. It also demonstrates why the model must not self-certify gardening decisions: four of the five filed issues failed independent read-back.

### Finding audit

- **Retain, but hold: #8588.** The top-level CLI routing shape is real: 2,157 Go files and about 436K lines live in `cmd/fak`, four arbitrary top-level dispatch switch chunks remain in `main.go`, and `scoreRoutes` is valid table-driven prior art. The issue is now `needs-triage` / `triage-only` until its historical bug claims and conversion size are independently witnessed.
- **Close: #8589.** Empty directories are not tracked by Git, so reaping them cannot be a durable repository change. The sampled “micro-packages” were mostly deliberate recent consolidation leaves with multiple consumers. The candidate also combined unrelated filesystem cleanup and package architecture in one issue.
- **Close: #8590.** The named scratch producers contain 1,281 Go files, not about 22,500; 21,186 of the observed files were in a different producer. `_scratch` contributes zero packages to `go list ./...`, so the issue also misattributed the shared-tree build-overlay requirement.
- **Close: #8591.** The search has six unfinished-work-comment hits tagged to #5331 rather than four, and parent #5331 is already closed as shipped. Introducing a production anchor merely to single-source leftover-marker prose would preserve stale work rather than resolve it.
- **Close: #8592.** The run did not preserve the alleged complete list of about 16 scripts, making its done condition irreproducible. Several named examples have real consumers: scheduled-task XML, launch scripts, docs, a plist, task inventory, or test companions.

The issue comments carry the exact independent evidence used to retain or close each candidate. Final gardening disposition: **one held for human-quality scoping, four closed as unsupported or mis-scoped**.

### Operating recommendation

Classify Ox Alpha for medium-scale repository review as **RECIPE**, not an autonomous default:

1. Use a read-only first pass; do not let the same trajectory both discover and file.
2. Require a complete machine-readable finding packet with commands, counts, candidate paths, and dedupe queries before issue creation.
3. Route every candidate through `fak-dev issue contract` with strict witness, scale, project-work, and born-routed checks.
4. Independently read back each factual claim and parent state before live gardening.
5. Apply labels/milestone at creation; an unlabeled issue is held, not dispatchable work.
6. Keep a separate reconciliation pass empowered to retain, rewrite, or close the model's candidates.

Ox Alpha is useful here as a cheap scout and synthesizer. Its observed failure mode is confident conflation: a true global smell was combined with stale parent state, non-versioned residue, incomplete inventories, and incorrect attribution. The right integration is therefore **model proposes; deterministic tools measure; an independent pass gardens**.
