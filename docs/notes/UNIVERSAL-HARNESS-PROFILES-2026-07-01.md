# Universal harness profiles — declarative config + auto-rotation for any sub-harness

_Design note for epic #1951. Landed with C1 (#1952). Current-state claims are witnessed
against the files/lines cited (read 2026-07-01)._

## The two planes

`fak guard -- <agent>` wraps any coding harness and routes every proposed tool call
through the kernel gateway. Two different things get decided when it does:

1. **The wire plane — already abstracted.** Which upstream protocol the traffic is
   proxied on (Anthropic Messages / OpenAI chat / OpenAI Responses). This is the
   `internal/agent` `Provider` adapter layer and `internal/modelroute` `ProviderKind`.
   A profile *references* this plane; it never reimplements a wire.

2. **The harness-as-a-process plane — NOT abstracted (this epic).** Which harness this
   executable is, how its child is pointed at the gateway, where its credential lives,
   and how its accounts rotate. Today every one of those facts is a separate `switch`
   keyed on the same underlying fact — the harness's identity:

   - detection → wire: `guardDetectProvider` (`cmd/fak/guard_provider.go:54`)
   - wire → URL: `guardDefaultBaseURL` (`:91`)
   - repoint: `if guardIsCodex` (`guard_codex.go:50`) / `case "claude"`
     (`guard_provider.go:303`, `guard_mcp.go`, `guard_precompact.go`, `guard_stophook.go`)
   - credential: `resolveAnthropicOAuthToken` (`guard.go`) vs
     `resolveCodexSubscriptionCredential` (`guard_codex_oauth.go:80`)
   - account home + rotation: `internal/accounts`, **Claude-only**
     (`Home`/`Identity`/`RotationPlan`/`ProjectSettings`)

`HarnessProfile` lifts that one fact into a first-class declared value, so guard reads a
descriptor instead of re-deciding it in each subsystem.

## The `HarnessProfile` shape

One descriptor per harness (built-ins shipped for claude / codex / openai-generic;
user-declarable in config as of C6). Defined in the pure leaf `internal/harnessprofile`
(tier: foundation — no `cmd/` or `internal/accounts` import):

| Field | Axis | Generalizes |
|-------|------|-------------|
| `Names []string` | **detect** — executable base names that select the profile | `guardAgentBaseName` + `guardDetectProvider` cases |
| `Wire` + `DefaultBaseURL` | **wire** — referenced adapter + its public base | `guardDetectProvider` / `guardDefaultBaseURL` |
| `Repoint []RepointMechanism` | **repoint** — how the child is pointed at the gateway | `if guardIsCodex` / `case "claude"` |
| `Credential CredentialSource` | **credential** — where the live token lives + its shape | the two resolvers |
| `ConfigHomeGlob` | **account home** — config-home dir convention | `accounts.Home` globs |
| `Identity IdentityKind` | **rotation identity** — which bucket-key reader to run | `accounts.Identity.AccountKey()` |

### The closed `RepointMechanism` set

A harness is pointed at the gateway by one or more of exactly three mechanisms — the
three that exist in guard today. The set is **closed**; C6 config validation rejects
anything else.

- `env` — inject the wire's base-URL env var(s) (`guardInjectedEnv`:
  `ANTHROPIC_BASE_URL`, or `OPENAI_BASE_URL` + `OPENAI_API_BASE`). **Always-on**:
  `guardInjectedEnv` fires for every provider, so every profile declares `env`.
- `cli-config` — Codex's `-c model_providers.fak.*` overrides
  (`installGuardCodexConfig`). Codex does not reliably honor `OPENAI_BASE_URL` for a
  custom upstream, so this is its real repoint.
- `settings-file` — Claude's `--settings` hooks (PreCompact/Stop) + `--mcp-config`
  self-query registration.

Built-in encodings (pinned by `TestRepointEncodesTodaysWiring`):

| Profile | Wire | Repoint |
|---------|------|---------|
| claude | anthropic | `env`, `settings-file` |
| codex | openai-responses | `env`, `cli-config` |
| openai-generic (opencode/aider/hermes) | openai | `env` |

### Credential + rotation identity are DECLARED, not executed here

The leaf is pure: it names the credential shape (`CredentialEnvKey` /
`CredentialOAuthFile`) and the identity reader (`IdentityClaude` / `IdentityCodex` /
`IdentityEnvKey` / `IdentityNone`) as data. The account model (C4, `internal/accounts`)
switches on `IdentityKind` to run the actual file read — that is where the I/O lives, so
`internal/harnessprofile` can stay a foundation leaf that a profile-consumer imports
without pulling in guard.

## Config surface (C6)

`HarnessProfile` overrides load from a `harnesses` table in `dos.toml` (the repo already
parses `dos.toml` for lanes/tiers, so a new file is not introduced). Precedence follows
the repo convention **flags > env > file > defaults**: built-ins are the defaults, a user
entry with the same detect-name overrides/extends. Load validates against the closed
vocabularies (`Wire.Valid`, `RepointMechanism.Valid`) with `DisallowUnknownFields`
discipline (mirrors `internal/policy/policy.go:220`), failing loud on an unknown wire or
mechanism. `fak guard --dump-harness-profiles` prints the merged, resolved set.

## Honest fences (what this is NOT)

- **Not a new wire layer.** Profiles reference the shipped `internal/agent` adapters.
- **Not auto-subscription-auth for every harness.** Codex ChatGPT-subscription upstream
  still needs its own increment (different host + `ChatGPT-Account-Id` header, per the
  `guard_codex_oauth.go` fence). Profiles make the *seam* uniform, not the upstream.
- **Not "every harness rotates on day one."** Ship the model + the two harnesses with
  real account models (claude, codex) first; opencode/aider/hermes are
  declared-but-thin (`IdentityEnvKey` / `IdentityNone`) until each has a credential/
  identity reader.
- **Behavior-preserving spine.** C2/C3 leave existing `fak guard` behavior byte-identical
  (existing guard tests green); the descriptor only *drives* what the switches did.

## Increment map

- **C1 (#1952)** — this note + `internal/harnessprofile`: the schema, the closed
  `RepointMechanism` set, and the built-in registry that encodes the current tables.
  `Lookup` reproduces `guardDetectProvider` for every known agent.
- **C2 (#1953)** — `guardDetectProvider` / `guardDefaultBaseURL` delegate to the registry.
- **C3 (#1954)** — one repoint dispatcher reads `profile.Repoint` instead of
  `if guardIsCodex` / `case "claude"`.
- **C4 (#1955)** — `internal/accounts` keys on a profile so `~/.codex*` homes enter the
  same rotation pool; Claude byte-identical.
- **C5 (#1956)** — `fak guard` consults the wrapped harness's rotation pool and rotates
  off a walled / `STALE_CRED` bucket (the headline behavior).
- **C6 (#1957)** — user `HarnessProfile` overrides in config, no Go.

Dependency order: **#1952 → { #1953, #1954, #1955, #1957 }; #1955 → #1956.**

## Related

- #1725 — harness-agnostic *routing* (the serving/wire win); this epic is the *config +
  rotation* complement.
- #620 — session-control-state epic; same "lift an implicit, reconstructed thing into a
  first-class declared value" pattern.
