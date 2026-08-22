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
