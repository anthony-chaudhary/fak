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

On an M3 or later Mac with 24 GB of memory or more and macOS 26.4 or later,
fak v0.58.0 or later can also run Inkling-2-Mini on the GPU by name:

```sh
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # v0.58.0
fak run inkling2-mini
```

Or build from a clone of `main`: `git clone --depth 1
https://github.com/anthony-chaudhary/fak.git`, then `go build -o fak ./cmd/fak`
(needs Go 1.26+).

The first run downloads about 14.6 GB. On an Apple Silicon Mac with 24 GB or
more, `fak run` with no model picks Inkling-2-Mini as well. Inkling-2-Mini is
in v0.58.0. GitHub's Latest release is still v0.56.0, so `install.sh` delivers
v0.56.0 unless you set `FAK_VERSION=0.58.0`. Smaller Macs and other
platforms keep the Qwen coding models. The [Mac guide](docs/fak/mac-local-models.md#inkling-2-mini-the-first-run-default)
has the requirements and the server commands.

For build requirements, platform verification and other model sizes, see the
[local model setup guide](GETTING-STARTED.md#local-model-setup).

[Connect your coding agent](docs/integrations/README.md) · [Hardware benchmarks](docs/benchmarks/README.md) · [How fak compares with llama.cpp](docs/native-inference-goal.md) · [Documentation](docs/index.md)

[Apache-2.0 license](LICENSE)

<!-- readme-verified: 2026-11-24 vs VERSION 0.57.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-11-24 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py + /refresh-readme -->
