<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak

fak is a local coding runtime.
Run a model on your own computer and keep your prompts local. Connect your
coding agent through fak’s gateway to allow reads while blocking writes.

[![fak benchmark sweep: Inkling-2-Mini decode and prefill on an RTX 5080 and decode on an M3 Pro, then two kernel checks, and where fak still trails](visuals/61-hero-benchmark-sweep.png)](docs/benchmarks/README.md)

On one RTX 5080, fak’s own engine decodes Inkling-2-Mini at 119.9 tok/s,
level with the fastest llama.cpp fork at 119.5. That is parity, not a lead:
fak’s number comes from its in-process bench and the fork’s from `llama-bench`.
Prefill is where fak pulls ahead: with 4-bit expert weights and activations
(NVFP4 W4A4, the default on an `sm_120f` build from `main`), it reads a
512-token prompt at 5861 tok/s against 3012 for the fork, 1.95x. That is the
server’s prompt rate; fak’s first visible token still arrives later than the
fork’s. At 1,946 tokens of context fak still trails the fork on decode, 89.0 vs
113.8 tok/s. The [benchmark index](docs/benchmarks/README.md) links the
receipts and the other comparisons.

## Try fak

[Install fak for your platform](INSTALL.md#local-model-setup), then open a
terminal.

On Apple Silicon, `fak up` probes unified memory, picks a model tier, and
opens a chat prompt:

```sh
fak up
```

On Windows x64 and Linux x64, `fak up` is not available (it is the Apple
Silicon provisioner). Run the small SmolLM2 model in-kernel instead. The first
run downloads it:

```sh
fak run smollm2 "What input breaks def first(xs): return xs[0]? Write a test that catches it."
```

Look for a test that passes an empty list and checks for `IndexError`.

On an M3 or later Mac with 24 GB of memory or more and macOS 26.4 or later,
fak v0.58.0 (pre-release) can also run Inkling-2-Mini on the GPU by name:

```sh
go install github.com/anthony-chaudhary/fak/cmd/fak@v0.58.0
fak run inkling2-mini
```

Or build from a clone of `main`: `git clone --depth 1
https://github.com/anthony-chaudhary/fak.git`, then `go build -o fak ./cmd/fak`
(needs Go 1.26+).

The first run downloads about 14.6 GB. On an Apple Silicon Mac with 24 GB or
more, `fak run` with no model picks Inkling-2-Mini as well. Inkling-2-Mini is
in v0.58.0. v0.58.0 is still a GitHub pre-release, so `install.sh`
installs an older release unless you set `FAK_VERSION=0.58.0`. Smaller Macs and other
platforms get smaller model tiers instead. The [Mac guide](docs/fak/mac-local-models.md#inkling-2-mini-the-first-run-default)
has the requirements and the server commands.

For the commands and their main flags, see the [CLI reference](docs/CLI.md).

For build requirements, platform verification and other model sizes, see the
[local model setup guide](GETTING-STARTED.md#local-model-setup).

For the version history, see [CHANGELOG.md](CHANGELOG.md).

[Connect your coding agent](docs/integrations/README.md) · [Hardware benchmarks](docs/benchmarks/README.md) · [How fak compares with llama.cpp](docs/native-inference-goal.md) · [Documentation](docs/index.md)

[Apache-2.0 license](LICENSE)

<!-- readme-verified: 2026-11-25 vs VERSION 0.58.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-11-25 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py + /refresh-readme -->
