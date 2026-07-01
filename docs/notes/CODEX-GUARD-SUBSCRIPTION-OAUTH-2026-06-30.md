# Wiring `fak guard -- codex` to the Codex ChatGPT-subscription OAuth token

**Date:** 2026-06-30
**Status:** increment 1 (resolver) code-complete; increments 2–6 are the plan below.
**Goal:** let `fak guard -- codex` hold a **Codex ChatGPT-subscription** credential
upstream (the `codex login` OAuth token), the way `fak guard -- claude` already holds a
Claude Pro/Max subscription OAuth token — instead of requiring `OPENAI_API_KEY` (API
billing), which is the only Codex auth guard wires today.

This is the named follow-on the Codex guard code already fences honestly:
`cmd/fak/guard_codex.go` says *"A `codex login` ChatGPT subscription is NOT yet wired
through guard the way the Claude Pro/Max subscription is."* This doc turns that fence into
a checkable increment plan.

---

## 1. The lesson from the Claude path (what to copy)

The Claude subscription path (`cmd/fak/guard.go`, `guard_child.go`, `internal/agent/adapters.go`)
works in five moves. Each has a Codex analogue:

| # | Claude move | Where | Codex analogue |
|---|---|---|---|
| A | **Resolve** the OAuth token from disk (env → `.credentials.json` → `.oauth-token`) | `resolveAnthropicOAuthToken` | Read `<codex-home>/auth.json` → `tokens.access_token` + `account_id` |
| B | **Pin** it upstream + re-read per request for rotation | `pinUpstream` + `apiKeyFunc` in `cmdGuard` | Same shape: Codex refreshes `auth.json` itself, so re-read per turn |
| C | **Pick the auth scheme** by token shape | `anthropicAdapter.Headers` (`IsAnthropicOAuthToken` → `Bearer` + `oauth-2025-04-20` beta) | Responses wire needs `Bearer` **+ `ChatGPT-Account-Id` header** |
| D | **Ignore the child's own key**; the gateway holds the real one | `PinUpstreamCredential` + placeholder `ANTHROPIC_API_KEY` | Placeholder `OPENAI_API_KEY`; gateway holds the OAuth token |
| E | **Default to subscription** unless an API key is explicitly named | `resolveGuardUpstream` anthropic branch | New codex branch keyed on `guardIsCodex` + no explicit `--api-key-env` |

## 2. Why Codex is harder than Claude (the load-bearing difference)

Claude's OAuth token goes to the **same host** (`api.anthropic.com`) — only the auth
*scheme* differs (Bearer + a beta header), and fak's anthropic adapter already switches
scheme by token prefix. So the Claude wiring needed **no new upstream and no new header
seam**.

Codex's ChatGPT-subscription token is different on **two axes** at once (verified
2026-06-30 against the OpenAI Codex auth docs and the `codex-rs` auth model):

1. **Different upstream.** The subscription path does NOT hit `api.openai.com/v1`. It hits
   the **ChatGPT backend** `https://chatgpt.com/backend-api/codex` on the **Responses**
   wire. So when this credential is held, guard's resolved upstream base URL must repoint
   there — not the `guardDefaultBaseURL("openai-responses")` = `api.openai.com/v1` default.
2. **Extra required header.** The backend requires `ChatGPT-Account-Id: <account_id>`
   beside the bearer. Drop it and it returns **401/403**. fak's current Responses adapter
   (`openAIResponsesAdapter.Headers`) emits only `Authorization: Bearer` and has **no seam**
   for a per-request extra header — only the Anthropic wire has `UpstreamBeta`
   (`internal/agent/stream.go` `upstreamCall.headers()` gates the merge on
   `ProviderAnthropic`).

These two facts are why this is a multi-increment change and not a one-line default flip.

### `auth.json` shape (ChatGPT / subscription mode)

```json
{
  "OPENAI_API_KEY": null,
  "auth_mode": "chatgpt",
  "tokens": {
    "id_token": "<JWT — encodes chatgpt_account_id under the https://api.openai.com/auth claim>",
    "access_token": "<OAuth bearer sent upstream>",
    "refresh_token": "<Codex refreshes with this; fak does NOT>",
    "account_id": "<ChatGPT-Account-Id header value (codex-rs placement)>"
  },
  "last_refresh": "<RFC3339; ~8-day stale window>"
}
```
Location of `account_id` has moved across Codex versions (top-level vs `tokens.` vs only in
the `id_token` JWT), so the resolver reads all three. Token + account id are a **matched
pair** and must come from the same file — a mismatched account id is a documented 401 cause.

## 3. Increment plan (each shippable + checkable on its own)

- **[1] Credential resolver — DONE (this change).**
  `cmd/fak/guard_codex_oauth.go` + `_test.go`. Pure, read-only; nothing calls it yet, so no
  live-path change (ships ahead of its caller like `codexMemoryBackend` did). Extracts
  `access_token` + `account_id` + `auth_mode` from `<codex-home>/auth.json`
  (CODEX_HOME-aware via `resolveCodexHome`). **Witness:** `go test ./cmd/fak -run Codex`
  (parse, JWT-claim fallback, API-key-mode refusal, CODEX_HOME).

- **[2] Responses adapter extra-header seam.** Add a generic per-request upstream-header
  carrier to `SampleParams` (e.g. `UpstreamHeaders map[string]string`) applied in
  `upstreamCall.headers()`, OR a Responses-specific `ChatGPT-Account-Id` field. Keep it a
  no-op off the Responses wire. **Witness:** an `httptest` upstream asserting the request
  carries `ChatGPT-Account-Id` (mirrors `adapters_test.go`'s Bearer/beta assertions).

- **[3] Subscription upstream base URL.** When the codex credential is held, resolve the
  upstream base to `https://chatgpt.com/backend-api/codex` (a `guardCodexSubscriptionBaseURL`
  const) and provider `openai-responses`. **Witness:** unit test on the resolver branch.

- **[4] `resolveGuardUpstream` codex branch.** Mirror the anthropic branch: when
  `guardIsCodex(agentName)` and no explicit `--api-key-env`, resolve the subscription
  credential, set `pinUpstream`, `apiKey=access_token`, stash the account id, and repoint
  the base (increment 3). Fall back to today's API-key path when there is no subscription
  login (the `parseCodexSubscriptionCredential` error already reports `auth_mode`).
  Add `apiKeyFunc` re-reading `auth.json` per request for rotation (Codex rewrites it,
  exactly like Claude Code rewrites `.credentials.json`). **Witness:** `guard_child`-style
  unit test asserting pin + source, with the account id threaded to the gateway.

- **[5] Child placeholder + gateway pin.** Inject a placeholder `OPENAI_API_KEY` into the
  Codex child (so its own credential check passes) and make the gateway ignore it on this
  path (the OpenAI-wire twin of `PinUpstreamCredential`; today that pin is Anthropic-only in
  `anthropicUpstreamCredential`). The Codex `-c model_providers.fak.*` override already
  points Codex at the gateway; `env_key` stays `OPENAI_API_KEY` and the placeholder
  satisfies it. **Witness:** the 4-check end-to-end proof from `docs/integrations/CLAUDE.md`
  re-run against `fak guard -- codex` (a `/v1/responses` 200 that is only possible because
  the gateway swapped in the held OAuth token).

- **[6] Banner + honest fences.** Update `printGuardCodexNote` to say "Codex ChatGPT
  subscription (OAuth from `<auth.json>`, account-id header set)" when pinned, and keep the
  API-key note otherwise. Update `docs/integrations/openai-codex.md` (the "API-key billing
  today" paragraph) once [5] lands. **Do not** claim subscription support until [2]–[5] are
  green end-to-end.

## 4. Honest fences (do not over-claim)

- fak does **not** refresh the token (no `refresh_token` flow). It relies on Codex's own
  proactive refresh + refresh-on-401 rewriting `auth.json`, which the per-request re-read
  (increment 4) then picks up — the same division of labor as the Claude path.
- The ChatGPT backend endpoint/headers are an **observed** integration surface, not a
  documented-stable public API. Increment 5 must be witnessed by a live 200, and the fence
  in the banner/doc must say "observed" until then.
- Never log the access token or the account id (redact) — same rule the Claude path follows.

## Source alignment (checked 2026-06-30)

- Codex CLI auth / `auth.json` fields, ChatGPT-mode, refresh lifecycle:
  <https://developers.openai.com/codex/auth> and
  <https://developers.openai.com/codex/auth/ci-cd-auth>
- ChatGPT-Account-Id passthrough + `chatgpt.com/backend-api/codex` Responses wire (prior art
  discussion): <https://github.com/chopratejas/headroom/issues/773>

## fak-side references

- Claude lesson: `cmd/fak/guard.go` (`resolveAnthropicOAuthToken`, `apiKeyFunc`),
  `cmd/fak/guard_child.go` (`resolveGuardUpstream`, `buildGuardChild`),
  `internal/agent/adapters.go` (`anthropicAdapter.Headers`, `AnthropicOAuthBeta`).
- Codex seam: `cmd/fak/guard_codex.go` (the honest fence this doc closes),
  `cmd/fak/guard_codex_oauth.go` (increment 1), `internal/gateway/responses.go`,
  `internal/agent/adapters.go` (`openAIResponsesAdapter`).
