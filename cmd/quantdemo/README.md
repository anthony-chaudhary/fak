# Portable quantized-model comparison demo

`quantdemo` is a small, neutral interoperability spine. It runs one redistributable
quantized model through the same pinned native runtime either directly (**without fak**)
or via fak's OpenAI-compatible gateway (**with fak**). The claim is deliberately narrow:
fak composes with this runtime and artifact. This is **not** a quality, latency, throughput,
memory, or universal quantization-winner claim.

## Deterministic offline witness

```sh
go run ./cmd/quantdemo -selfcheck
```

This uses no network, model, GPU, or mutable local state. It validates four contract fixtures
and prints the immutable pins. Unknown format/version returns `ABSTAIN/UNKNOWN_FORMAT`; an
unlisted runtime returns `DELEGATE/RUNTIME_NOT_PINNED`; and a known format/runtime with an
unwitnessed quantization returns `REFUSE/COMBINATION_NOT_LISTED`. There is no silent fallback.
With a warm Go build cache the selfcheck completes in under one second; an initial run may take
longer while Go compiles the package. Its output is deterministic and byte-identical across
re-runs, with the exact render locked to [EXAMPLE-OUTPUT.md](EXAMPLE-OUTPUT.md) by a Go test.

## What you see

Each `PASS` line names an asserted part of the compatibility contract: the four typed fixture
decisions, the immutable artifact and runtime pins, and the deliberately narrow composability
claim. The repository-wide [claim ledger](../../CLAIMS.md) documents the evidence labels used by
fak; this demo does not turn its offline fixtures into live quality or performance evidence.

## Immutable live pins

Print machine-readable pins with `go run ./cmd/quantdemo -pins`.

| Layer | Pin | License |
|---|---|---|
| artifact | `bartowski/SmolLM2-135M-Instruct-GGUF@09816acd5d99df7be770d85ea30822623dab342c`, `SmolLM2-135M-Instruct-Q4_K_M.gguf`, 105,454,432 bytes, SHA-256 `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d` | Apache-2.0 (model card metadata) |
| format/recipe | GGUF v3, `Q4_K_M` | artifact license applies; this demo does not invent an artifact format |
| delegated runtime | `llama.cpp@b6500+ga7a98e0fffed` (tag commit `a7a98e0fffed794396b3fbad4dcdbbc184963645`) | MIT; pinned LICENSE SHA-256 `e562a2ddfaf8280537795ac5ecd34e3012b6582a147ef69ba6a6a5c08c84757d` |

The model is downloaded at run time and is not redistributed by this repository. Review the
upstream model card/license before use; the recorded license is provenance, not legal advice.

## Live with/without-fak witness

Prerequisites: `git`, CMake, a C++ compiler, and the repository's `fak` binary. CPU is enough;
a sanctioned compute node may be used when the workstation cannot host the runtime.
Keep downloads/builds outside the repository or under `_scratch/issue-6262`.

```sh
scratch="${TMPDIR:-/tmp}/fak-quantdemo"
mkdir -p "$scratch"
git clone --depth 1 --branch b6500 https://github.com/ggml-org/llama.cpp.git "$scratch/llama.cpp"
cmake -S "$scratch/llama.cpp" -B "$scratch/llama.cpp/build" -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
cmake --build "$scratch/llama.cpp/build" --config Release -j 2 --target llama-server
curl -fL 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf' -o "$scratch/model.gguf"
printf '%s  %s\n' 2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d "$scratch/model.gguf" | sha256sum -c -
"$scratch/llama.cpp/build/bin/llama-server" -m "$scratch/model.gguf" --host 127.0.0.1 --port 8081 -ngl 0 -c 512
```

In separate terminals, compose that unchanged endpoint through fak and run the witness:

```sh
fak serve --addr 127.0.0.1:8080 --base-url http://127.0.0.1:8081 --model SmolLM2-135M-Instruct

go run ./cmd/quantdemo -live -model "$scratch/model.gguf" \
  -runtime 'llama.cpp@b6500+ga7a98e0fffed' \
  -direct-url http://127.0.0.1:8081/v1/chat/completions \
  -fak-url http://127.0.0.1:8080/v1/chat/completions
```

Success means both paths returned an OpenAI-compatible response to the same deterministic
request, and the output records each body independently. Text equality is not asserted because
runtime scheduling can vary. No timing is compared: the gateway adds capabilities and an extra
hop, while a direct call is the honest alternative when those capabilities are not needed.


