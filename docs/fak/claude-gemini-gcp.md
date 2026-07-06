---
title: "claude-gemini-gcp: Gemini 3.5 Flash from your GCP account, fronted by the kernel"
description: "One preset command points the real Claude Code CLI at Gemini 3.5 Flash on GCP Vertex AI, with the fak kernel adjudicating every tool call. It is a tier-2 dispatch seat that needs no GPU node — only your existing GCP creds. What is proven on any host vs what needs live GCP access."
---

# `claude-gemini-gcp` — Gemini 3.5 Flash from your GCP account

> **Audience.** Operators who already have GCP credentials and want to run Claude Code
> against **Gemini 3.5 Flash** as a cheap, fast **tier-2** seat, with the fak kernel in
> front of every tool call. By the end you can install the preset, dial Vertex AI, and
> know exactly what is proven on any host versus what needs live GCP access.

One preset command runs the real Claude Code CLI on **Gemini 3.5 Flash served by GCP
Vertex AI**, with the fak kernel adjudicating every tool call. It is the same
`dogfood-claude` openai backend as [`claude-glm-gcp`](claude-glm-gcp.md), pointed at a
Google-**managed** model instead of a self-hosted GPU node — so there is **no VM to stand
up**, only your existing GCP creds.

```
 ┌───────────────────┐   /v1/messages   ┌────────────────────────┐  /chat/completions  ┌──────────────────────────┐
 │ claude-gemini-gcp │ ───────────────▶ │ fak serve (the kernel) │ ──────────────────▶ │ Vertex AI OpenAI-compat  │
 │   (Claude Code)   │ ◀──── SSE ─────── │ openai backend, adjud. │ ◀────────────────── │ gemini-3.5-flash (GCP)   │
 └───────────────────┘                   └────────────────────────┘                     └──────────────────────────┘
        ▲ ANTHROPIC_BASE_URL=loopback fak          every tool call crosses the      Authorization: Bearer <GCP token>
        │ FAK_GEMINI_GCP_PROJECT=<id>              kernel floor first
```

Because Vertex is a managed API, the two halves of the GLM story collapse to one: there
is no "stand the node up" step. You provide a GCP project and a bearer token; fak proxies
straight to Vertex's OpenAI-compatible endpoint and adjudicates the round trip.

## The one command

Install the launchers once, then point the preset at your project:

```bash
./scripts/dogfood-claude.sh --install            # installs claude-gemini-gcp (+ fak, the other presets)
```
```powershell
.\scripts\dogfood-claude.ps1 --install           # Windows: claude-gemini-gcp.cmd + fak.exe
```

Then set your project and a bearer token, and go:

```bash
export FAK_GEMINI_GCP_PROJECT=<your-gcp-project>          # or reuse GCP_PROJECT
export FAK_GEMINI_GCP_KEY="$(gcloud auth print-access-token)"   # a GCP access token
claude-gemini-gcp --probe "say pong"             # one witnessable headless turn
claude-gemini-gcp                                 # interactive Claude Code on Gemini 3.5 Flash
```

### What the preset is

`claude-gemini-gcp` is the same launcher as `fak-dogfood`; its name selects
`FAK_DOGFOOD_PRESET=gemini-gcp`. The preset defaults to:

| setting | value |
|---|---|
| backend | `openai` (fak proxies straight to the Vertex `/chat/completions`) |
| model-server URL | built from `FAK_GEMINI_GCP_PROJECT` + `FAK_GEMINI_GCP_LOCATION` (default `global`), or set `FAK_GEMINI_GCP_BASE_URL` directly |
| model id | `google/gemini-3.5-flash` (`FAK_GEMINI_GCP_MODEL` overrides) — every Claude tier maps onto it |
| auth | `FAK_GEMINI_GCP_KEY` holds the bearer; fak sends it as `Authorization: Bearer` (`fak serve --api-key-env`) |
| tool-result compatibility | `FAK_OPENAI_TOOL_MESSAGES_AS_TEXT=1` by default for this preset, because Vertex's OpenAI-compatible Gemini route rejects Claude tool-result role messages unless fak serializes them as text |

The base URL fak builds is the Vertex OpenAI-compat endpoint:

```
https://aiplatform.googleapis.com/v1beta1/projects/<project>/locations/global/endpoints/openapi
```

For a regional location, the host is `https://<location>-aiplatform.googleapis.com/...`.

fak's openai backend appends `/chat/completions`, which is the Vertex chat path. Override
any default with the normal `FAK_DOGFOOD_*` env vars (`FAK_DOGFOOD_BASE_URL` overrides the
built URL; `FAK_DOGFOOD_MODEL` the id; `FAK_DOGFOOD_API_KEY_ENV` the bearer env var name).

### On the bearer token

A `gcloud auth print-access-token` token is short-lived (about an hour). For a longer
session, refresh it and re-export `FAK_GEMINI_GCP_KEY`, or point the preset at an
Application-Default-Credentials proxy that mints tokens on demand. If you would rather use
an AI-Studio-style long-lived key against `generativelanguage.googleapis.com`, set
`FAK_GEMINI_GCP_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai` and put
that key in `FAK_GEMINI_GCP_KEY` — but the default, and the point of this preset, is your
**GCP** account via Vertex.

## Tier 2, and a dispatch seat

Gemini 3.5 Flash is registered as a **tier-2** model — the lightweight-work tier, next to
GLM-5.2 — in the fleet-accounts taxonomy (`internal/fleetaccounts/modelTierFromName`,
mirrored in `tools/fleet_accounts.py`). So a dispatch seat whose model resolves to
`gemini-3.5-flash` (in any id shape, including the Vertex `google/gemini-3.5-flash`) is
routed as tier 2: `fak dispatch` sends **gardening / maintenance / light** work to it and
reserves tier-1 seats for the hard reasoning.

Two ways to make it a seat:

- **Provider-account roster** (portable, reviewable): the account +
  binding are in [`examples/model-accounts.example.json`](../../examples/model-accounts.example.json)
  as `gemini-gcp-vertex` → `gemini-flash` (`google/gemini-3.5-flash`). See
  [`docs/model-accounts.md`](../model-accounts.md).
- **Config-home seat**: pin a Claude config-home account to tier 2 with an
  `account_profiles` override (`model_tier: 2`, `model: google/gemini-3.5-flash`) in
  `tools/_registry/accounts_policy.json`, and launch that seat with
  `FAK_DOGFOOD_PRESET=gemini-gcp`.

## Monitoring effectiveness

Effectiveness of a Gemini seat is read from the same surfaces every dispatch seat reports
into — this preset adds a seat, not a new metric:

- **Dispatch telemetry** — `fak dispatch tick`/`wave` record per-seat launches, and
  `fak accounts status --json` / `fak fleet-accounts roster` surface `can_serve`, login
  status, and the seat's resolved tier.
- **Trajectory audit** — the `/trajectory-audit` skill and `fak` trajectory tooling score
  a seat's turns (task completion, tool-call correctness, retries) from its session
  journal.
- **Provider-cache / cost lenses** — the gateway `/metrics` counters and the cache-value
  rollup show what the seat actually spent and reused.

A dedicated *Gemini-seat* effectiveness witness (a live tier-2-vs-tier-1 task-quality
comparison) needs live GCP turns and is tracked as a follow-on — see
[What is proven here](#what-is-proven-here).

## What is proven here

This follows the repo's serving-honesty boundary: the **mechanism** lands and is
witnessed on any host; the **live model turn** is gated on live GCP access (a project with
Vertex AI enabled + a valid bearer token), which is not exercised from the implementing
host — the same gate as [`claude-glm-gcp`](claude-glm-gcp.md).

| Item | Witness | Status |
|---|---|---|
| The `gemini-gcp` preset resolves to fak's openai backend at the Vertex OpenAI-compat endpoint with model `google/gemini-3.5-flash` | `go test ./cmd/fak -run TestClaudeGeminiGCP` (bash + PowerShell launchers) | ✅ proven on any host |
| The bearer is threaded as `Authorization: Bearer` from `FAK_GEMINI_GCP_KEY` (the mac-preset auth path) | same test — `DEFAULT_UPSTREAM_API_KEY_ENV="FAK_GEMINI_GCP_KEY"` → `fak serve --api-key-env` | ✅ proven on any host |
| Gemini tool-result messages are serialized as text by default, so a Claude Code tool turn can continue after the tool result | same test — `DEFAULT_OPENAI_TOOL_MESSAGES_AS_TEXT="1"` / `$PresetOpenAIToolMessagesAsText = '1'`; live witness `experiments/agent-live/dogfood-claude-gemini-gcp-tool-canary-texttools-20260706T075408Z.json` ran one `Bash` command and returned seat counts | ✅ mechanism proven on any host; live witness captured on this host |
| Gemini 3.5 Flash is a **tier-2** model in the fleet-accounts taxonomy | `go test ./internal/fleetaccounts -run TestModelTierFromNameGeminiFlashIsTier2` + `tools/fleet_accounts_test.py::test_model_tier_from_name_gemini_flash_is_tier2` | ✅ proven on any host |
| The dispatch seat is in the provider-account roster (`gemini-gcp-vertex` → `gemini-flash`) | `examples/model-accounts.example.json` round-trips `--accounts-check` | ✅ proven on any host |
| A **live Gemini 3.5 Flash text turn** through the preset against Vertex | `experiments/agent-live/dogfood-claude-gemini-gcp-20260706T075103Z.json` (`result: pong`) | ✅ captured on this host; needs live GCP access elsewhere |
| A **live tier-2 effectiveness comparison** of the seat vs a tier-1 seat on real dispatch work | needs live GCP turns + dispatch runs | ⏳ GCP-gated (follow-on) |

The wire, tier, and tool-result compatibility are wired and witnessed from source on any
host. This host also captured live text and tool-use canaries against Vertex; another host
still needs live GCP access before it can repeat them.

## Troubleshooting

- **`OpenAI-compatible endpoint not reachable` / 401 / 403** — the bearer expired or the
  project lacks Vertex access. Re-run `export FAK_GEMINI_GCP_KEY="$(gcloud auth
  print-access-token)"` and confirm the project has the Vertex AI API enabled and the
  account has `roles/aiplatform.user`.
- **404 on the model** — the region does not serve `gemini-3.5-flash`, or the id needs the
  `google/` publisher prefix. Keep the default `google/gemini-3.5-flash`; the preset defaults
  to `FAK_GEMINI_GCP_LOCATION=global` because the global endpoint is the broadest Gemini
  OpenAI-compatible surface.
- **HTTP 400 after a tool call** — the provider accepted the tool call but rejected the
  follow-up tool-result role. Keep the preset default `FAK_OPENAI_TOOL_MESSAGES_AS_TEXT=1`;
  if you override it to `0`, tool-using Claude Code turns can fail after the first tool result.
- **The launcher dies asking for a project** — set `FAK_GEMINI_GCP_PROJECT` (or
  `GCP_PROJECT`), or pass a full `FAK_GEMINI_GCP_BASE_URL`.

## Refs

- `scripts/dogfood-claude.sh` / `.ps1` — the launcher + the `gemini-gcp` preset
- [`claude-glm-gcp.md`](claude-glm-gcp.md) — the sibling preset (self-hosted GLM-5.2 on a GCP GPU node)
- [`docs/model-accounts.md`](../model-accounts.md) — the account switcher (the dispatch-seat roster)
- `internal/fleetaccounts/fleetaccounts.go` · `tools/fleet_accounts.py` — the tier taxonomy (Gemini 3.5 Flash → tier 2)
- [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) — the general one-command dogfood launcher
