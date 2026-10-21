# dflash2bench: experimental physical diagnostic

This standalone command compares ordinary greedy decoding with the native
DFlash2 request path on the **same authenticated quantized Qwen3.8-27B target**.
It is a preliminary paired measurement, not a serving benchmark or a claim of
speedup. The current five-layer draft uses host BF16 streaming math and target
tap readbacks; it may be slower than ordinary decoding. The default target verifier runs serial native Step calls, with restore/replay
on partial acceptance. Explicit panel4/panel8 selects the source-reviewed native
block verifier; its physical qualification and throughput remain unmeasured.

## Required inputs and runner

- A compatible explicitly selected backend. The current execution-owner provider
  is Vulkan on Linux/Windows with cgo. An unavailable backend is an error; there
  is no implicit CPU fallback.
- The independently trusted ordered GGUF shard manifest, its shard-set SHA-256,
  the embedded tokenizer identity, and the renderer identity used for the prompt.
- An immutable local stock sidecar:
  `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`,
  3,848,817,896 bytes, SHA-256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
  This is the published identity enforced by the loader, not a claim that a
  particular machine already has those bytes.
- An authorized, capacity-admitted runner. This command does not download
  artifacts, reserve device memory, or establish runner authorization.
  See [the Vulkan build prerequisites](../../docs/compute/vulkan-linux.md).

The target-spec JSON uses the fields of `agent.DFlash2TargetSpec`:

```json
{
  "Shards": [
    {"path": "/absolute/target-00001-of-00018.gguf",
     "size": 0, "raw_file_sha256": "REPLACE_WITH_TRUSTED_RAW_FILE_DIGEST"}
  ],
  "ExpectedShardSetSHA256": "REPLACE_WITH_INDEPENDENT_ORDERED_MANIFEST_DIGEST",
  "TokenizerID": "sha256:REPLACE_WITH_AUTHENTICATED_TOKENIZER_ID",
  "RendererID": "qwen-chatml-v1:no-thinking",
  "SidecarPath": "/absolute/model.safetensors"
}
```

Replace the illustrative shard list with every real ordered shard and its actual
size. The manifest digest is SHA-256 of the loader's canonical JSON encoding of
that ordered array; it is **not** one shard's raw-file hash. Do not invent the
trusted digest by hashing arbitrary unverified files at invocation time.
All target and sidecar files must remain immutable until the process exits;
retained readers do not prevent in-place file mutation.

The prompt file is a nonempty JSON array of already rendered token IDs (maximum
8192). Render/tokenize it with the supplied renderer and authenticated tokenizer
ahead of time. The benchmark does not invent a chat template, add BOS, render
text, or apply string stops. Both arms exclude the same authenticated EOS and
stop IDs before commit.

## Prepare immutable inputs offline

`dflash2bench prepare` authenticates the complete ordered GGUF shard set before
reading its embedded tokenizer and using the same in-kernel renderer as the
benchmark target. It requires `fak.dflash2.trusted-target.v1`, containing
`model_id`, ordered `shards`, an independently trusted
`expected_shard_set_sha256`, explicit `expected_renderer_id`, and the absolute
future `sidecar_path`. `expected_tokenizer_id` may be omitted: after every GGUF
byte matches the trusted size and raw SHA-256 declarations, preparation derives
the embedded operational tokenizer identity and emits it. If supplied, it must
match exactly. The independent trust root remains the pinned shard declaration;
the command never creates trust by hashing arbitrary observed files.

The `fak.dflash2.prompt-request.v1` input contains a versioned `prompt_id`, the
same `model_id` and `renderer_id`, `messages`, optional `tools`, and
`max_output_tokens`. Unknown fields and renderer drift refuse. Example:

```sh
dflash2bench prepare \
  --target-declaration /absolute/trusted-target-declaration.json \
  --prompt-request /absolute/versioned-prompt-request.json \
  --target-spec-out /absolute/prepared-target-spec.json \
  --prompt-ids-out /absolute/prepared-prompt-ids.json \
  --receipt-out /absolute/preparation-receipt.json
```

All paths in the target declaration must remain immutable through preparation
and the later benchmark run. Output files are created new and never overwrite an
existing path. The source-bound `fak.dflash2.preparation.v1` receipt hashes both
input files and both emitted benchmark inputs, and records the actual tokenizer,
renderer, model/config, and clean source revision identities.

`PREPARED` means input preparation only. This command does not open or authenticate
the DFlash2 sidecar, initialize Vulkan, use a GPU, run inference, or provide
physical evidence. The admitted hardware controller must separately authenticate
the pinned sidecar and preserve the prepared files before execution.

## Build and run

After building the existing Vulkan shim/shaders as documented, use the same
native link configuration. For example, with the admitted build directory in
`build`:

```sh
CGO_ENABLED=1 CGO_LDFLAGS="-L$build" go build -tags vulkan -o "$build/dflash2bench" ./cmd/dflash2bench
FAK_VULKAN_SPIRV="$build/spirv" "$build/dflash2bench" \
  --backend vulkan --target-spec /absolute/trusted-target.json \
  --prompt-ids /absolute/prerendered-prompt-ids.json \
  --tokens 64 --order plain-first --timeout 30m > plain-first.json
```

Run `--order draft-first` separately to expose order effects. Both sessions begin
with cache length zero, but device weights and the OS page cache may warm between
arms. The command does not call a warmup "cold" and does not infer a stable speedup
from one pair. Use an external resource supervisor and hard deadline: the command
timeout is cooperative and cannot preempt a blocked native device call.

A default non-Vulkan build can provide `--help`; it must refuse a requested
unavailable Vulkan backend rather than measure the CPU under that name. No stock
model run or hardware throughput result is included with this source.

## Read the result

The JSON report records the Go build information (including VCS settings when
embedded), exact trusted artifact/config identities, actual backend/tier/caps,
prompt-file hash, initial cache size, and exact output token IDs.

It separates:
- common target authentication/conversion/load time;
- session construction and each request's complete sidecar open/authentication;
- fresh-prefix prefill;
- TTFT from request start through the first successful token commit;
- decode wall time, including the final emitted token's commit;
- draft-proposal wall time, which includes host math and shared target
  embedding/head calls;
- the exact target Verify method's wall time;
- total Prepare and its residual work, Commit/replay/draft append, and finalization.

Every DFlash2 request currently rehashes the entire 3.85 GB sidecar. This cost is
included in request TTFT and request-total wall time, and shown separately from
decode throughput. Startup-owned verified sidecar reuse is a possible later
optimization, not implemented by this command.

Proposed draft tokens exclude the mandatory target anchor. Requested positions,
verified draft tokens, committed draft tokens, and committed target anchors have
separate counters. Stop-before-commit can reduce the committed prefix. The
optional per-call PrepareTimed observation uses the same locked math/state path;
ordinary Prepare does not read the clock.

The existing always-on device-work and host-fallback counters are sampled without
enabling a timestamp profile. Missing counters are unavailable, not zero.
`SubmitWaitObserved=false` means that field does not cover all relevant waits.
Readback-byte counters are aggregate; they do not isolate tap traffic. A registered
kernel profile is recorded because profiling can change the execution regime.

A ratio is printed only after both arms and target cleanup succeed, with exact
emitted-token and committed-position parity and nonzero output. A mismatch,
cancellation, incomplete cleanup, panic, or Goexit cannot produce a passing pair.
An uncertain owner cannot produce success or start another arm. Returned errors
exit nonzero; a cleanup Goexit can leave backend/background goroutines alive until
the required external hard supervisor terminates them. No unsafe backing is
force-retired to permit continued execution. Do not embed this direct-session diagnostic in a
concurrent gateway: direct Session users are outside the gateway lease contract.

The upstream draft mathematics license remains
[the published MIT notice](../../internal/model/dflash2_reference_LICENSE.txt).


## Explicit target-panel selection

The default remains --draft-mode host --target-panel serial. To measure the
same host drafter with native target verification, explicitly select
--target-panel panel4 or --target-panel panel8. Width includes the mandatory
target anchor, so panel4 permits at most three draft positions. Unsupported
capabilities or selected-operation errors refuse; they do not fall back to
serial. Bootstrap still performs serial teacher forcing.

For each selected panel, run both --order plain-first and --order draft-first.
Compare its exact-token-gated results with the corresponding serial runs.
The new completed_target_verification entries copy actual verifier path,
row count, operations and decode steps after successful verification. They
are not device-clock measurements or evidence of end-to-end speedup. A
requested selection alone is not a selected-work receipt.

The draft arm now acquires the same target/domain lease used by serving.
diagnostic_gate_session_sidecar_open_and_verification includes that gate,
fresh Session construction and full sidecar hash/open; separate session_open
and sidecar_open fields are unavailable (zero) for this owning draft route.
The per-request 3.85 GB sidecar verification and host draft math remain.

--draft-mode resident-mlp returns an explicit admission-unavailable error before
target/model/sidecar acquisition. Although a separate resident MLP primitive
exists, the complete physical target/snapshot/panel/staging memory recipe has
not been admitted. This command never allocates resident matrices and then
labels a host request accelerated. No new environment flag changes the route.

The typed integration and panel primitives remain experimental; their exact
composed bytes still require compilation and execution. Use the same external hard supervisor,
artifact identities, cleanup checks and parity gate documented above.
