# Codex session adapter

`internal/codexsession` launches the installed Codex app server as a local stdio child and projects its typed JSON-RPC notifications into `fak.harness.run/v1`. The browser continues to consume the existing harness reducer; provider wire objects never become renderer contracts.

## Prerequisites and launch

Call `codex --version` and pass that exact value in `codexsession.Config.Version`. Set `Workspace` to the repository root and construct the adapter with a run ID and event sink. `Run(ctx, text)` launches `codex app-server`, initializes it, starts one ephemeral thread and one turn, and streams semantic message, tool, usage, error, and completion events. `Interrupt()` sends the typed `turn/interrupt` request for the active turn.

The supported protocol is the app-server v2 method/notification set shipped by the installed Codex CLI: `initialize`, `thread/start`, `turn/start`, `turn/interrupt`, `item/*`, `thread/tokenUsage/updated`, and `turn/completed`. A missing version is rejected so unsupported installations cannot be presented as negotiated. Transport is always labeled `stdio://`; receipts and completion are labeled `engine=codex` because execution is Codex-backed, not fak-native.

## Safety, failures, and rollback

The workspace is normalized to an explicit absolute child working directory. The adapter does not scrape the TUI or infer protocol from prose. Child startup, malformed JSON, RPC errors, unexpected EOF, failed turns, and cancellation are returned or emitted semantically. To roll back, stop selecting this adapter; the public harness protocol and existing fak-native producer are unchanged.

The focused seeded trace witness is `TestAdapterProjectsTypedAppServerStream`. It runs a deterministic fake app server over real stdio, captures ordered public events, and proves incremental assistant deltas, tool activity, usage, completion, engine/transport/version labeling, and absolute workspace binding. `TestInterruptUsesTypedRPC` captures the cancellation request.


## Approval authority

Codex command and patch server requests remain typed JSON-RPC requests. The adapter
correlates thread, turn, item, and request identity, asks the injected fak capability
floor before projecting a semantic `approval.requested`, and accepts a response only
with the active input lease/principal and execution epoch. Structural denial, unknown
request kinds, timeout, disconnect, and stale or duplicate responses all decline.
Receipts name the Codex sandbox/approval policy separately from fak's additional
capability floor; journals contain scrubbed summaries and identities, never raw
command output, environment values, or credentials.

## Experimental protocol compatibility gate

Before `Run` starts the Codex app-server, callers that opt into the experimental
adapter provide a `CompatibilityEnvelope` and the `CompatibilityReceipt` loaded
from a tested fixture. The gate binds binary version, negotiated protocol, and
the fixture's SHA-256 digest, then compares the authority-bearing method set.
Unknown observation events remain an adapter/rendering concern; unknown or
renamed approval methods fail closed before process or tool execution.

Refresh a tested version by capturing a scrubbed generated schema in
`testdata/conformance`, running the package conformance tests, and shipping the
new fixture with its receipt binding. Roll back by restoring the previously
tested Codex binary/protocol pair; diagnostics report all three binding values.

Promotion evidence from `gen/next` is a release gate running these fixtures
against the installed Codex minimum and current versions. Demote or retire the
adapter if those receipts cannot be reproduced or authority drift repeatedly
requires bypasses. This gate assumes the local generated schema completely
names every authority-bearing request; an out-of-band authority channel
invalidates that assumption and must keep the adapter experimental.
