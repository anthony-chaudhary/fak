# Local Router Quickstart (`hybrid` profile)

`examples/local-router` is a starter hybrid routing setup for `fak serve`.

> **TL;DR:** Run `fak serve` with `accounts.json` and `manifest.json` to keep `private` local-only while falling back to cloud on `default`.

Configured bindings:

- `default`: routes to `local-gpu` (`http://127.0.0.1:8080/v1`) and falls back to `openai` (`gpt-4o-mini`).
- `private`: routes to `local-gpu` with `residency_floor: "local_only"` (fails closed rather than spilling to cloud).
- `frontier`: routes to `anthropic` (`claude-sonnet-4-20250514`) and falls back to `openai` (`gpt-4o`).

You can also generate this setup (or `local-only` / `multi-cloud` profiles) into any directory with:

```bash
fak route init --profile hybrid --out-dir .fak-router
```

## 3-Step Run

1. Export credentials for the cloud fallback/frontier accounts (and ensure your local OpenAI-compatible server is listening on `127.0.0.1:8080`):

   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. Start `fak serve` with the roster, manifest, and tool-call guard:

   ```bash
   fak serve --route-accounts examples/local-router/accounts.json --route-manifest examples/local-router/manifest.json --guard
   ```

3. Inspect live routing and failover state:

   ```bash
   curl -s http://127.0.0.1:8080/v1/router/status
   ```
