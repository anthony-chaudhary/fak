# Captured quantized-model selfcheck output

Command, run from the repository root:

```sh
go run ./cmd/quantdemo -selfcheck
```

Captured stdout:

```text
PASS fak-quantdemo/1 fixtures=4
PASS pins model=SmolLM2-135M-Instruct-Q4_K_M.gguf sha256=2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d license=Apache-2.0
PASS runtime=llama.cpp@b6500+ga7a98e0fffed license=MIT license_sha256=e562a2ddfaf8280537795ac5ecd34e3012b6582a147ef69ba6a6a5c08c84757d
PASS typed unknown-format=ABSTAIN unknown-runtime=DELEGATE unsupported-combination=REFUSE
CLAIM composability-only: same pinned artifact/runtime, direct and through fak; no quality or performance winner
```
