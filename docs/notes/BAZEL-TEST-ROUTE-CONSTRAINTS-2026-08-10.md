# Bazel platform-constraint test-route witness (#6401)

## Verdict

Bazel platform constraints represented the same four probes as
`internal/testroute`: native-allowed selected `native`, Windows with WSL selected
`wsl`, CI-only selected `ci`, and unavailable selected no artifact. The witness
was 4/4 accurate. The zero-artifact unavailable result is expected and is not
ranked as a failed route.

This is an equivalence witness for route selection, not a claim that Bazel can
execute the repository's Go, WSL, or GitHub Actions routes. It adds no root Go
dependency, nested module, workflow, runner, or CI input.

## Pinned inputs

- Host: Windows control point, PowerShell, x86-64.
- Bazelisk: `v1.28.1`, reused from the shipped #6371 fixture; npm package
  `@bazel/bazelisk`, package file SHA-256
  `45ffea1a7a0971eb0f2bb27019002c2cc851ef784da09facd8234fe742d26706`.
- Bazel: `9.2.0`, selected by `bazel/.bazelversion`; binary SHA-256
  `5fc2f2805b8c697a54732558576938d06bab63aa0f9b6610cc01d2cae0388705`.
- Module: `affected_tests_diamond` version `1.0`; `MODULE.bazel` SHA-256
  `65de63871349de217cb5802bd07b6ddceeabe0642e68fa35cf3434cd69016d8a`.
- Lock: committed `MODULE.bazel.lock` SHA-256
  `38731963ff6d7df650a7355090c4388b7218e064bc75f839531902dc92f98023`.
  Bazel's built-in toolchain analysis resolves the lock-pinned `platforms`
  repository; the fixture declares no `bazel_dep`.
- Configuration: `routing/BUILD.bazel` declares one constraint setting, four
  constraint values, four platforms, and one selected `filegroup` effect.

The exact version read-back was:

```text
Bazelisk v1.28.1
Build label: 9.2.0
```

## Commands and effects

Run from `examples/affectedtests-comparison/bazel`. `<root>` is an ignored
scratch output root; the committed fixture contains no generated Bazel state.

```powershell
$env:BAZELISK_HOME = '<pinned-cache>'
$env:USE_BAZEL_VERSION = '9.2.0'
bazelisk --output_user_root=<root> version

bazel.exe --batch --output_user_root=<root> cquery //routing:route_effect `
  --platforms=//routing:probe_native_allowed --output=files --repository_disable_download
bazel.exe --batch --output_user_root=<root> cquery //routing:route_effect `
  --platforms=//routing:probe_windows_wsl --output=files --repository_disable_download
bazel.exe --batch --output_user_root=<root> cquery //routing:route_effect `
  --platforms=//routing:probe_ci_only --output=files --repository_disable_download
bazel.exe --batch --output_user_root=<root> cquery //routing:route_effect `
  --platforms=//routing:probe_unavailable --output=files --repository_disable_download
```

| Probe | Expected effect | Read-back | Correct | Applicable work |
|---|---:|---:|---:|---:|
| native-allowed | `routing/native.route` | `routing/native.route` | yes | 1 |
| Windows + WSL | `routing/wsl.route` | `routing/wsl.route` | yes | 1 |
| CI-only | `routing/ci.route` | `routing/ci.route` | yes | 1 |
| unavailable | zero output lines | zero output lines | yes | 0 |

Accuracy was 4/4 (1.0), with zero false routes. Applicable-route precision and
recall were both 3/3 (1.0). The unavailable case remains zero/non-applicable.

## Measurements and provenance

The process-tree sampler from shipped #6371 polled every 20 ms and recorded wall
time, summed process-tree CPU, and peak aggregate working set. Each route used
Bazel `--batch`; a single measured setup command populated the lock-backed
repository cache, and every scored route then succeeded with repository downloads
disabled.

| Phase/probe | Wall (s) | CPU (s) | Peak RSS (bytes) | Network |
|---|---:|---:|---:|---|
| setup fetch + native smoke | 13.2416281 | 13.890625 | 294,203,392 | bytes unknown; Bazel exposed no transfer counter |
| native-allowed | 2.7201349 | 7.718750 | 336,965,632 | 0 repository-download bytes by enforced flag |
| Windows + WSL | 4.7144866 | 7.890625 | 336,433,152 | 0 repository-download bytes by enforced flag |
| CI-only | 3.9369750 | 7.578125 | 278,994,944 | 0 repository-download bytes by enforced flag |
| unavailable | 3.4025771 | 7.687500 | 307,515,392 | 0 repository-download bytes by enforced flag |

The four scored routes totalled 14.7741736 wall seconds and 30.875 CPU seconds,
for 0.2707427 routes/second; maximum sampled peak RSS was 336,965,632 bytes.
These are single observed cold batch-mode runs, not distributional estimates.

Operator elapsed time from first fixture write through metric capture was
208.7768936 seconds. Setup/operator time is reported, not monetized. Incremental
local monetary cost was observed as USD 0.00: no paid runner or service was
invoked. Total monetary cost for this witness is therefore USD 0.00 within that
scope; hardware, electricity, and labor rates are not known and are not inferred.

## Independent read-back contract

An independent verifier must read the committed files rather than this narrative,
rerun all four `cquery` commands with downloads disabled, assert the exact three
artifact lines and the empty unavailable stdout, recompute the aggregate metrics
from the machine-readable JSON, and verify the pushed commit from `origin/main`.

Machine-readable evidence: `affected-test-route-bazel-6401.json`.
