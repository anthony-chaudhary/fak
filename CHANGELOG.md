# Changelog

All notable changes to fak are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The GitHub release
page for each version is the full record.

## [Unreleased]

## [0.59.0] - 2026-11-26

- Inkling-2 can reuse a bounded prefix cache across requests when enabled.
- The Metal backend reuses prefill chunks across requests and uses speculative
  decoding by default for the resident target when a draft is available.
- Published CPU, Metal, and Vulkan archives include SHA-256 checksums. CPU and
  CUDA container images are available for this version.

[Full release notes and downloads](https://github.com/anthony-chaudhary/fak/releases/tag/v0.59.0)

## [0.58.0] - 2026-11-25 (pre-release)

This pre-release did not publish its CUDA container or aggregate `SHA256SUMS`.

### Added

- Inkling-2-Mini runs on the GPU by name (`fak run inkling2-mini`) on Apple
  Silicon Macs with an M3 or later chip, 24 GB or more of memory and macOS 26.4
  or later.
- A bounded cache of resident Inkling-2 sessions, so interleaved conversations
  each reuse their own prompt prefix.
- Opt-in shared-prefix prefill for Inkling-2 cohorts
  (`FAK_INKLING2_COHORT_SHARED_PREFIX`, off by default).

### Changed

- Inkling-2 serving streams content deltas as soon as the decoded text cannot
  begin a tool call.

## [0.56.0] - 2026-10-12

Minor release covering compute, model, ggufload and computetrace.
