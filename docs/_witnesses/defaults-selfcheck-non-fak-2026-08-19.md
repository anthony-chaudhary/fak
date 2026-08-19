# Integrated defaults selfcheck in a non-FAK repository — 2026-08-19

**Verdict:** `fak doctor defaults-selfcheck` creates a disposable non-FAK repository and passes one deterministic behavioral packet over the default repository tools, harness profiles, context transforms, and calibrated virtual-cache decisions without a provider key.

Captured command:

```text
go run ./cmd/fak doctor defaults-selfcheck --json
```

The packet proves:

- all six bounded repository tools execute through the owned kernel loop, while an out-of-tree read is denied before engine dispatch;
- Caveman and Ponytail inject through both supported harness seams (Claude append-system-prompt and Codex developer instructions);
- live Anthropic transforms shorten compactible history, elide a superseded Read, defer a cold tool behind ToolSearch, and place a stable cache anchor;
- measured minimum-prefix, cached-read, 5m/1h write-price, and retention constants change their live decision/accounting seams;
- normalized cache-read usage reaches the provider-neutral family window;
- five launch postures are captured for `agent`, guarded Claude, guarded Codex, native serve, and passthrough serve;
- OpenAI cold-tool deferral is explicitly `unsupported`, not promoted to a false pass, because no witnessed OpenAI-compatible discovery mechanism exists.

Structured packet: [`defaults-selfcheck-non-fak-2026-08-19.json`](defaults-selfcheck-non-fak-2026-08-19.json).

Paid-provider augmentation remains separately labeled: this deterministic acceptance command does not claim a live provider call or GPU execution.
