# Local hardware benchmark receipt

This self-contained, **standard-library-only Go example** addresses [issue #10421](https://github.com/anthony-chaudhary/fak/issues/10421). It runs one benchmark command that **you explicitly select**, inventories the local setup, writes a portable JSON receipt, verifies that receipt offline, and renders a deterministic GitHub submission packet. It never uploads anything and never changes or falls back to another engine.

## Quick start

From the repository root:

```sh
cd examples/local-hardware-benchmark
go run . inventory
go run . run --out receipt.json -- <your benchmark command> <its arguments>
go run . verify receipt.json
go run . submit receipt.json
```

`run` streams the child process output to your terminal. It records the raw output's SHA-256 digest and a privacy-scrubbed, size-limited copy in the receipt. A nonzero child exit still produces a receipt, then returns an error.

The example does not decide what constitutes a valid performance or quality claim. Use an explicit fak-native benchmark command and the operating envelope required by that benchmark. **There is no automatic engine selection, substitution, or fallback.**

## Copy/paste paths by hardware

The commands below deliberately leave the benchmark arguments visible. Replace the placeholder with the exact fak-native benchmark command you intend to measure; confirm available benchmark verbs with `fak bench --help` in your checkout.

### Apple Silicon / Metal

```sh
cd examples/local-hardware-benchmark
go run . inventory
go run . run --out apple-metal.json -- fak bench <explicit-fak-native-benchmark> --engine metal
go run . verify apple-metal.json
go run . submit apple-metal.json
```

Inventory uses `system_profiler` and `xcrun metal --version` when available. It does not infer that a benchmark should use Metal merely because Apple hardware is present.

### NVIDIA / CUDA

```sh
cd examples/local-hardware-benchmark
nvidia-smi
go run . inventory
go run . run --out nvidia-cuda.json -- fak bench <explicit-fak-native-benchmark> --engine cuda
go run . verify nvidia-cuda.json
go run . submit nvidia-cuda.json
```

Inventory uses `nvidia-smi` and `nvcc --version` when available. Install the NVIDIA driver/toolchain appropriate for your machine before running a CUDA benchmark.

### AMD / ROCm

```sh
cd examples/local-hardware-benchmark
rocminfo
go run . inventory
go run . run --out amd-rocm.json -- fak bench <explicit-fak-native-benchmark> --engine rocm
go run . verify amd-rocm.json
go run . submit amd-rocm.json
```

Inventory uses `rocminfo` when available. ROCm support varies by operating system and GPU; the example reports what the installed tool exposes rather than claiming compatibility.

### AMD / Vulkan

```sh
cd examples/local-hardware-benchmark
vulkaninfo --summary
go run . inventory
go run . run --out amd-vulkan.json -- fak bench <explicit-fak-native-benchmark> --engine vulkan
go run . verify amd-vulkan.json
go run . submit amd-vulkan.json
```

`vulkaninfo` is recorded as a toolchain fact. The example does not silently replace ROCm with Vulkan or Vulkan with another backend.

### Intel accelerator / oneAPI-SYCL

```sh
cd examples/local-hardware-benchmark
sycl-ls
go run . inventory
go run . run --out intel-oneapi.json -- fak bench <explicit-fak-native-benchmark> --engine oneapi
go run . verify intel-oneapi.json
go run . submit intel-oneapi.json
```

Inventory uses `sycl-ls` when available. The exact engine spelling remains an operator choice and must match the fak command being measured.

### CPU-only

```sh
cd examples/local-hardware-benchmark
go run . inventory
go run . run --out cpu.json -- fak bench <explicit-fak-native-benchmark> --engine cpu
go run . verify cpu.json
go run . submit cpu.json
```

No accelerator is required. An empty accelerator list means CPU-only or that optional probes were unavailable; it is not proof that the machine has no accelerator.

## Receipt contents and offline verification

Schema: `fak.local-hardware-benchmark.receipt/v1`.

The receipt records:

- UTC start/finish timestamps and elapsed milliseconds;
- the scrubbed argument vector selected by the operator;
- child exit status;
- raw combined stdout/stderr byte count and SHA-256 digest;
- a scrubbed output excerpt (default maximum 32 KiB, configurable with `--max-output`);
- OS, architecture, normalized CPU description, physical memory, accelerators, and available toolchain facts;
- `fak version` output facts when `fak` is available;
- current Git repository revision when run inside a checkout;
- Go runtime version;
- a SHA-256 integrity seal covering every receipt field except the seal itself.

`go run . verify receipt.json` uses only local bytes and the Go standard library. It performs no network access. Any field edit causes verification to fail with an integrity mismatch. The integrity digest detects accidental or deliberate changes; it is **not a signature** and does not prove who ran the benchmark.

## Privacy behavior

Before writing the receipt, the example replaces common sensitive values with `[REDACTED]` and home-directory prefixes with `[HOME]`. It scrubs:

- common token, password, secret, API-key, authorization, and credential assignments/flags;
- bearer tokens;
- Windows, macOS, Linux, and UNC user-home paths;
- hostname, computer-name, username, user, and machine-ID assignments;
- the current process home directory if discoverable.

Hardware text is normalized to remove whitespace noise and reported CPU clock suffixes. Volatile driver versions are not included in accelerator model names, though concise toolchain facts may include installed versions.

**Review the JSON and generated submission body yourself.** Pattern-based scrubbing cannot recognize every proprietary model name, unusual secret format, command argument, filesystem layout, benchmark output, or other identifying datum. Use `--max-output 0` to omit the output excerpt while retaining its digest and byte count:

```sh
go run . run --max-output 0 --out private.json -- fak bench <explicit-fak-native-benchmark> --engine <explicit-engine>
```

The receipt file is created with owner-only permissions where the operating system honors POSIX permission bits.

## Submission behavior

```sh
go run . submit receipt.json > submission.txt
```

`submit` first verifies the receipt, then prints:

1. a stable Markdown issue body; and
2. a deterministic `github.com/.../issues/new` URL with the title, body, and `benchmark` label prefilled.

It does **not** invoke `gh`, open a browser, make a network request, create an issue, or upload the receipt. Open the URL only after reviewing the packet. Attach or paste the JSON separately if the destination requests it; the URL contains the summary, not the full receipt or output excerpt.

## Tests

```sh
go test ./examples/local-hardware-benchmark
```

Fixture-backed tests cover Apple, NVIDIA, AMD, and Intel hardware normalization; privacy scrubbing; receipt verification and tamper detection; and deterministic submission rendering.

## Limitations

- Hardware discovery is best-effort and depends on locally installed OS/vendor commands.
- Memory inventory reports physical memory, not free memory or device VRAM.
- The combined output digest preserves exact raw-byte identity, but stdout/stderr interleaving may reflect pipe scheduling performed by the child process and OS.
- The receipt does not judge benchmark comparability, quality, thermals, power state, driver suitability, or whether a command exercised the intended device. Inspect the command and benchmark-specific evidence.
- SHA-256 integrity is tamper-evident only when the expected digest is obtained from a trusted channel; it is not cryptographic attestation.
