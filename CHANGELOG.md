# Changelog

All notable changes to fak are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The GitHub release
page for each version is the full record.

## [Unreleased]

## [0.58.0] - 2026-11-25 (pre-release)

Pre-release, not the latest release. Its CUDA container and aggregate
`SHA256SUMS` did not publish, so `install.sh` and Homebrew deliver 0.56.0
until a later release is promoted.

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

## [0.56.0] - 2026-10-12 (latest release)

Minor release covering compute, model, ggufload and computetrace. This is the
release `install.sh` and the Homebrew formula deliver.
