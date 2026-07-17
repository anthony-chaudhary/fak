# Captured output

Start the browser demo:

```bash
go run ./cmd/qwen36codedemo
```

Startup output:

```text
qwen36codedemo: http://127.0.0.1:8154 (gateway http://127.0.0.1:8153, model Qwen3.6-27B-Q4_K_M)
```

A successful process remains running and serves `GET /healthz` with HTTP `200`; stop it with `Ctrl-C`. The page requires a compatible fak gateway for live model requests.
