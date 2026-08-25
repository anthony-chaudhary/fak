# Codex session adapter

`internal/codexsession` launches the installed Codex app server as a local stdio child and projects its typed JSON-RPC notifications into `fak.harness.run/v1`. The browser continues to consume the existing harness reducer; provider wire objects never become renderer contracts.

## Prerequisites and launch

Call `codex --version` and pass that exact value in `codexsession.Config.Version`. Set `Workspace` to the repository root and construct the adapter with a run ID and event sink. `Run(ctx, text)` launches `codex app-server`, initializes it, starts one ephemeral thread and one turn, and streams semantic message, tool, usage, error, and completion events. `Interrupt()` sends the typed `turn/interrupt` request for the active turn.

The supported protocol is the app-server v2 method/notification set shipped by the installed Codex CLI: `initialize`, `thread/start`, `turn/start`, `turn/interrupt`, `item/*`, `thread/tokenUsage/updated`, and `turn/completed`. A missing version is rejected so unsupported installations cannot be presented as negotiated. Transport is always labeled `stdio://`; receipts and completion are labeled `engine=codex` because execution is Codex-backed, not fak-native.

## Safety, failures, and rollback

The workspace is normalized to an explicit absolute child working directory. The adapter does not scrape the TUI or infer protocol from prose. Child startup, malformed JSON, RPC errors, unexpected EOF, failed turns, and cancellation are returned or emitted semantically. To roll back, stop selecting this adapter; the public harness protocol and existing fak-native producer are unchanged.

The focused seeded trace witness is `TestAdapterProjectsTypedAppServerStream`. It runs a deterministic fake app server over real stdio, captures ordered public events, and proves incremental assistant deltas, tool activity, usage, completion, engine/transport/version labeling, and absolute workspace binding. `TestInterruptUsesTypedRPC` captures the cancellation request.
