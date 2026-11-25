---
title: "Campaign record: Inkling-2-Mini on the 36 GiB M3 Pro"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak m3pro-inkling2-mini-metal (2026-11-24): Inkling-2-Mini on the 36 GiB M3 Pro."
---

# Inkling-2-Mini on the 36 GiB M3 Pro

The MacBook has 18 GPU cores and runs macOS 27.0.1 on arm64. All engines read the same Mainline_Support UD-Q3_K_XL weights. The checkpoint contains 24 weight layers, 12 recurrent layers and three passes through those recurrent layers, for 48 layer visits per token. The target and DFlash2 sidecar came from owned LAN storage.

| Model file | Bytes | SHA-256 |
| --- | ---: | --- |
| Inkling-2-Mini-UD-Q3_K_XL.gguf | 13,631,779,296 | e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639 |
| Inkling-2-Mini-DFlash2/model.safetensors | 989,975,616 | 217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee |

## Which Fak version has this

| | |
| --- | --- |
| Qualified source | public `main` at [1f31533f2](https://github.com/anthony-chaudhary/fak/commit/1f31533f2ccba844c2f03d176bfbb98772913754), committed 2026-11-24 |
| Latest tagged release | [v0.57.0](https://github.com/anthony-chaudhary/fak/releases/tag/v0.57.0), tagged 2026-11-13 from ad2f918a4. It **does not** contain the native Inkling-2 Metal runner. |
| First release with it | the first tag after v0.57.0 |
| Version string | A build from `main` between those tags still prints `0.57.0`. Check the `build:` line instead. |

The release installer (`install.sh`) and Homebrew install the latest tag, so until the first tag after v0.57.0 ships, build from `main`:

```sh
git clone --depth 1 https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak    # needs Go 1.26+ and cgo (Xcode command line tools)
./fak version   # expect "0.57.0" plus a "build:" line naming a main commit at or after 1f31533f2ccb
```

The commands below call the binary `fak`; from the clone, run `./fak` or put it on your `PATH`. `go install ...@main` is not a substitute: the Go module proxy timed out fetching `main` when this page was checked, and `...@latest` resolves to v0.56.0.

Then run the model by alias. On Apple Silicon this downloads the pinned target GGUF and the DFlash2 draft on first use and places the draft next to the GGUF. The alias, and its use as the default for a bare `fak run` on Macs with 24 GB or more, landed on `main` at 6f56c684d, after the qualified revision above, so use a current `main` build:

```sh
fak run inkling2-mini --effort none --timings --temp 0 --max-tokens 256 --quiet "What is 2 + 2? Reply with the number only."
```

The measurements on this page used an explicit path to a local copy of the same file, which still works:

```sh
fak run "$HOME/models/inkling2/Inkling-2-Mini-UD-Q3_K_XL.gguf" --effort none --timings --temp 0 --max-tokens 256 --quiet "What is 2 + 2? Reply with the number only."
```

Models: the target is published as `Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file` in [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The DFlash2 draft is [incoai/Inkling-2-Mini-DFlash2@4f9088ea](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2). Put the draft in an `Inkling-2-Mini-DFlash2/` directory next to the GGUF; Fak picks it up automatically. Check both files against the SHA-256 values above. This run used a 36 GB M3 Pro. The process peaked at about 16 GB.

Only this page and [perf-receipt-v2.json](perf-receipt-v2.json) are published in the public repository. The `receipts/` files, reproduction scripts and schedules linked below are kept in the maintainer archive, because they record local paths.

## Native Metal qualification

On this MacBook, the final native runner reached 97.5% to 107.4% of Splash 1.7.2’s decode rate on the two natural continuations. It reached 148.1% on repeated text, where verified prompt lookup can supply longer exact runs. Prefill ranged from 98.2% to 107.5% of Splash. These ratios describe warm, single-request throughput for this checkpoint, device and schedule.

Both engines read the same target and DFlash2 weights. The comparison ran four fresh processes in the order Splash, Fak, Fak, Splash. Each received the same 24 requests, including identical warmups and six measured requests per shape across the two processes. Splash reported zero prompt-cache hits on all requests. Fak reset the request state between prompts. Each cell reports the median observed tokens per second.

| Prompt, input / output tokens | Fak prefill | Splash prefill | Fak decode | Splash decode | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Capitals, 19 / 32 | 160.46 | 149.28 | 40.33 | 37.54 | 1.074× |
| Fibonacci, 20 / 32 | 145.25 | 143.21 | 55.50 | 54.92 | 1.011× |
| Repeated text, 128 / 32 | 320.74 | 323.77 | 177.05 | 119.58 | 1.481× |
| Repeated text, 512 / 1 | 458.75 | 467.37 | — | — | — |
| Capitals, 19 / 128 | 153.05 | 148.19 | 47.76 | 46.49 | 1.027× |
| Fibonacci, 20 / 128 | 145.56 | 148.09 | 63.57 | 65.21 | 0.975× |

For 128-token Fibonacci, Fak ranged from 58.68 to 67.90 decode tokens/s and Splash from 59.00 to 71.14. Their medians differ by 2.5%. The native verifier used a median of 21 cycles and Splash used 19. Individual samples, per-process medians and measured ranges are in [the paired summary](receipts/n-abba-summary.json) and [observed ranges](receipts/n-observed-ranges.json).

**Token agreement.** Each engine matched all 582 short-suite IDs and all 768 Fibonacci IDs against the respective target references. On 128-token capitals, Fak matched the independent Koosh continuation in three of six runs and Splash in two of six. Both engines first diverged at positions 42, 66 or 112, depending on the run. At position 112 both can choose Koosh’s fifth-ranked token within 0.237 nats. That choice falls outside the existing runner-up-only test. The target-only Fak control matched both 128-token references exactly. A physical test verified continuity when EOS or the generation limit lands inside a speculative block. [Oracle evidence](receipts/long-oracle-verdict.json) · [EOS and generation-limit test](receipts/n-eos-limit.log)

**Implementation.** Fak invokes the Apple GPU operators in its own process. It verifies candidates against the full target, retains the accepted target KV, short convolution and draft context on the GPU, and checks the resulting row mask against an independent host-side walk before exposing tokens. An ABBA comparison of the same driver with embedded source versus the upstream precompiled shader library found a 1.5–2.4% decode gain from the precompiled version and a 9.1% loss on fixed verification. Fak ships the source. [Shader comparison](receipts/native-source-vs-release-metallib.json) · [Device commit probe](receipts/device-commit-probe.json)

**Follow-up measurements.** For validation we held the verifier at 8 or 16 rows and used the same two 128-token prompts. The 16-row run matched both independent sequences exactly but dropped Fibonacci from 71.87 to 59.86 tokens/s in that trial. The 8-row run reached 72.27 tokens/s and still diverged on capitals. These single warmed observations support retaining adaptive sizing. [Fixed-width measurements](receipts/N-fixed-width-summary.json) · [Independent verdict](receipts/independent-judge.json) · [Paired-trial audit](receipts/independent-N-audit.json)

**Source and tests.** The qualified public revision is [1f31533f2](https://github.com/anthony-chaudhary/fak/commit/1f31533f2ccba844c2f03d176bfbb98772913754). All changed production sources match the benchmark build byte for byte. The committed tree additionally contains the passing physical EOS and generation-limit test. The public CLI, agent, loader and Metal package checks passed. The private repository’s full boundary-tool suite still has two earlier catalog failures, `TestNamesake_RealTreeClusterFleetProvenance` and `TestEveryPlatformDirHasCard`. The changed-path boundary and publication scans passed. [Build identity](receipts/n-build-identity.json) · [Focused checks](receipts/n-focused-tests.log)

**Installed CLI.** This Mac’s stamped Fak 0.57.0 reports public revision `1f31533f2ccb`. Automatic backend selection reproduced the 32-token raw capitals text. The following chat command returned `4` in 5.09 seconds, including loading. A 16-token limit ended before any visible answer. The first installed raw run took 21.99 seconds, including first-use shader compilation. Repeating the same command in a fresh process took 4.43 seconds. The DFlash2 directory sits alongside the GGUF. [Raw CLI output](receipts/installed-cli-capitals.json) · [Repeat run](receipts/installed-cli-capitals-repeat.json) · [Chat output](receipts/installed-cli-chat.json)

```sh
fak run "$HOME/models/inkling2/Inkling-2-Mini-UD-Q3_K_XL.gguf" --effort none --timings --temp 0 --max-tokens 256 --quiet "What is 2 + 2? Reply with the number only."
```

The standard [hardware receipt](perf-receipt-v2.json) records the device, model, kernel, argv and timing provenance. Lesson PHL-213-4cf8e5 records the width comparison. Reproduce the final table from the archived measurements in this directory.

```sh
python3 final_paired_report.py --schedule m-qualification.jsonl --target receipts/koosh-long-oracle.json \
  --splash receipts/npaired-a-splash.jsonl receipts/npaired-d-splash.jsonl \
  --fak receipts/npaired-b-fak.jsonl receipts/npaired-c-fak.jsonl --output /tmp/inkling-m3-paired.json
```

## Comparison

Mainline llama.cpp b12681 measured 18.9, 18.9 and 19.0 decode tokens/s for capitals, Fibonacci and the repeated 128-token prompt. Koosh0610/llama.cpp-inkling at cf5f8a7 measured 23.4, 23.2 and 23.3. Splash 1.7.2 with DFlash2 measured 39.5, 56.7 and 120.6 in the initial control. Each engine returned the same 32 token IDs on each prompt. The repeated prompt contains long verbatim matches and benefits from prompt lookup.

The paired control uses a fresh Splash process and a fresh Fak process on this Mac. Both run one request at a time and receive identical token IDs. Each of the four shapes runs once to warm the device and three times for measurement, in the order in [warm-matrix.jsonl](warm-matrix.jsonl). Every Splash request has a unique cache salt and reports zero matched prompt tokens. Each Fak request starts with empty logical token history, KV and convolution state. Report the median of each phase independently. Prefill throughput divides all input tokens by wall time through the first-token projection. Decode throughput divides the remaining 31 tokens by decode seconds. The 512-token case requests one output and measures only prefill. Cold process launch, weight loading and shader compilation are separate from warm throughput.

## Measured implementation changes

The first packed Metal implementation returned all 64 short-prompt oracle IDs. Warm decode was 7.3–7.6 tokens/s. Keeping state on the GPU raised the two-prompt paired medians to 18.60 and 19.83. Fusing controller projections raised those medians to 20.60 and 21.28 with the same IDs.

Fak's in-process source port of the Apache-2.0 Splash operators owns model loading, tokenization, generation, greedy acceptance and request state. The Apple Metal device compiles embedded shader source. A transaction test stages a bad continuation, accepts a prefix and discards the remaining rows. A second test follows a noncontiguous tree path and checks the following token against full-depth target inference.

The first fused-tree candidate matched 291 of 291 IDs across the 12 measured requests, including the three one-token prefill checks. Its medians against the paired Splash control follow.

| Input shape | Fak prefill | Splash prefill | Fak decode | Splash decode |
| --- | ---: | ---: | ---: | ---: |
| Capitals, 19 tokens | 130.76 | 144.10 | 31.59 | 37.20 |
| Fibonacci, 20 tokens | 133.20 | 134.18 | 55.87 | 52.79 |
| Repeated text, 128 tokens | 317.33 | 327.95 | 156.67 | 114.85 |
| Repeated text, 512 tokens | 461.72 | 458.09 | — | — |

The ordinary command selected native Metal with both explicit and automatic backend selection. Its raw capitals output matched the 32-token oracle byte for byte. With the chat template, effort none and a 256-token limit, the reply to “What is 2 + 2? Reply with the number only.” was “4”. On that fresh process the total time was 5.40 seconds, including model loading. Its maximum process footprint was 16.14 GB. The first run of the newly built executable paid 19.98 seconds at prefill for cold shader compilation. Warm throughput applies to the loaded model.

The raw measurements and binary and model identities are under [receipts](receipts). Splash source is pinned at [2afc7337](https://github.com/incoai/splash/tree/2afc7337aacc044722fa95e4cb3da75c281a19a8). The two llama.cpp references are [mainline b12681](https://github.com/ggml-org/llama.cpp/releases/tag/b12681) and [Koosh's Inkling fork](https://github.com/Koosh0610/llama.cpp-inkling/tree/cf5f8a770eb4ee7ee03e35f8975b35715dbd3536).

## Adaptive verification and held-out continuation

The next candidate reuses one idle session arena and resets its logical state between requests. The fused verifier combines short prompt matches with neural proposals and uses observed GPU time and acceptance to size the next block. Median session acquisition and teardown overhead fell from 42 ms to 1 ms. Against a second interleaved Splash control it reproduced all 291 short-suite output IDs.

| Input shape | Fak prefill | Splash prefill | Fak decode | Splash decode |
| --- | ---: | ---: | ---: | ---: |
| Capitals, 19 tokens | 151.48 | 145.96 | 37.89 | 35.74 |
| Fibonacci, 20 tokens | 141.22 | 131.03 | 54.56 | 51.90 |
| Repeated text, 128 tokens | 324.34 | 321.16 | 164.18 | 112.13 |
| Repeated text, 512 tokens | 428.87 | 448.31 | — | — |

Extending the two natural continuations to 128 tokens exposed a separate numerical and throughput question. The adaptive candidate matched 128 of 128 Fibonacci tokens, at 53.16 tokens/s against the initial Splash run's 72.91. Capitals first diverged at the 42nd generated token. The candidate chose token 160146, while Splash and both llama.cpp controls chose 151462 there. The same Fak binary with single-token target verification matched the two llama.cpp capitals continuations for all 128 tokens. Raw short-suite evidence alone cannot establish longer-run greedy equivalence.

Run the independent summary from this directory with `python3 paired_report.py receipts/splash-paired-b.jsonl receipts/fak-adaptive-tree-l.jsonl`. To collect a new matched Fak sample, copy [the benchmark driver](matched_fak.go.txt) under a temporary directory inside the public Fak module, build it there with Go, and supply the model path on the command line with [the JSONL trial schedule](warm-matrix.jsonl) on standard input. The equivalent [Splash collector](splash_matched.py) accepts the release binary, its Python package and resolved model directory and uses the same schedule.

## Wide verification

The wide verifier can combine seven neural predictions and up to 15 prompt-lookup proposals in a 16-row transaction. A longer verified prompt copy can use up to 32 rows. The private control and production runner both use the same public Go generation policy. All 291 measured short-suite tokens matched. The ordinary Metal CLI also reproduced the raw 32-token capitals text with 6.04 seconds elapsed, including loading and teardown.

| Input and output tokens | Fak prefill | Splash prefill | Fak decode | Splash decode |
| --- | ---: | ---: | ---: | ---: |
| Capitals 19 / 32 | 150.39 | 145.96 | 38.23 | 35.74 |
| Fibonacci 20 / 32 | 133.76 | 131.03 | 54.95 | 51.90 |
| Repeated text 128 / 32 | 322.99 | 321.16 | 161.94 | 112.13 |
| Repeated text 512 / 1 | 463.45 | 448.31 | — | — |
| Capitals 19 / 128 | 142.40 | 166.83 | 45.36 | 52.97 |
| Fibonacci 20 / 128 | 144.01 | 163.70 | 57.73 | 71.36 |

Each 128-output arm includes one warmup and three measured repeats. For Fibonacci, the three Fak and three Splash runs matched the same 128 Koosh target tokens. For capitals, one of three Fak runs and two of three Splash runs matched all 128 Koosh tokens. Two Fak repeats chose a separate space and then spelled “Brasilia” where Koosh chose “ Brasília”. At this first mismatch, output index 65, the chosen token was Koosh’s runner-up by 0.086 nats. Repairing the prefix and continuing found a second runner-up at index 111, 0.139 nats behind the top choice. Repairing that prefix reproduced the remaining 16 IDs. This meets the existing Inkling oracle’s two-reanchor, runner-up, 0.5-nat criterion. At index 111 one Splash run chose Koosh’s fifth-ranked token, 0.236 nats below its first. Exact-token and near-tie results are reported separately.

A target-only shape control forced the identical 40-token prefix into single rows or fully accepted multirow blocks. The next choice flipped with the block width despite zero rejected tokens. An independent state audit checked 59,424,768 bytes of convolution history and 12,874,752 bytes of retained KV values and scales over 31 commits. Each matched its source row exactly.

Reproduce the short and long comparisons from [the combined schedule](m-qualification.jsonl) with `python3 paired_report.py receipts/splash-paired-b.jsonl receipts/fak-wide-tree-m.jsonl` and `python3 long_report.py receipts/splash-paired-long.jsonl receipts/koosh-long-oracle.json receipts/fak-wide-tree-m.jsonl`.
