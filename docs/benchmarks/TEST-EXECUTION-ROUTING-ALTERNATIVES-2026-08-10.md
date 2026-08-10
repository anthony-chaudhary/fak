# Test-execution routing alternatives â€” 2026-08-10

Status: **INCOMPLETE**. Issue [#6353](https://github.com/anthony-chaudhary/fak/issues/6353) tracks real execution witnesses.

The same native-allowed, Windows-with-WSL, CI-only, and unavailable probes are routed to exact executable choices. Arms: fak native; GOOS-only baseline; fak + GitHub Actions; Go native; WSL wrapper; GitHub Actions routing; Bazel platform constraints. Completion requires route accuracy, latency/throughput, CPU/RSS/network, setup/operator time, total cost, pinned versions/configuration, commands, and independent read-back. Three Windows/amd64 local samples were 31.07, 30.70, and 27.19 ns/op (median 30.70 ns/op; zero allocations). Unavailable arms remain measurement-zero; no cross-system ranking exists yet.
