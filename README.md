<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak

fak is a local coding runtime.
Run a model on your own computer and keep your prompts local. Connect your
coding agent through fak’s gateway to allow reads while blocking writes.

## Try fak

[Install fak for your platform](INSTALL.md#local-model-setup), then open a
terminal. Use the same command on Windows x64, Linux x64, and Apple Silicon
Macs to start the coding model.

```sh
fak up --model 3B
```

The first run downloads a small coding model (about 1.9 GB). At `you>`, paste
`def first(xs): return xs[0]` and ask “What input breaks this function?
Write a test that catches it.” Look for a test that passes an empty list and
checks for `IndexError`. Type `/quit` to exit.

For build requirements, platform verification and other model sizes, see the
[local model setup guide](GETTING-STARTED.md#local-model-setup).

[Connect your coding agent](docs/integrations/README.md) · [Hardware benchmarks](docs/benchmarks/README.md) · [How fak compares with llama.cpp](docs/native-inference-goal.md) · [Documentation](docs/index.md)

[Apache-2.0 license](LICENSE)

<!-- readme-verified: 2026-11-24 vs VERSION 0.57.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-11-24 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py + /refresh-readme -->
