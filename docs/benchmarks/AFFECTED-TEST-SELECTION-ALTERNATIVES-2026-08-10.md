# Affected-test selection alternatives â€” 2026-08-10

Status: **INCOMPLETE**. Issue [#6371](https://github.com/anthony-chaudhary/fak/issues/6371) tracks real build-system witnesses.

The same diamond import graph with one changed leaf must select the leaf and every transitive importer while excluding an isolated package. Arms: fak native; changed-only baseline; fak + Go test; Bazel; Pants; Nx; Gradle. Failure attribution and flaky rerun classification remain separate package capabilities. Completion requires selection precision/recall, missed/extra packages, tests run, latency, CPU/RSS/network, setup/operator time, total cost, pinned graphs/configuration/commands, and independent read-back. Three Windows/amd64 samples were 362.4, 428.5, and 506.5 ns/op (median 428.5 ns/op; 224 B/op; 8 allocs/op). Unavailable arms stay measurement-zero; no cross-system ranking exists yet.
