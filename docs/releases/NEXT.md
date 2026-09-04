# fak vNext (targeting v0.47.0): Work in Progress

This document tracks in-flight work on `main` targeting the upcoming `v0.47.0` release.
It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.

- **Projected version:** `0.47.0` (`minor` bump)
- **Base release tag:** `v0.46.0`
- **Commits in flight:** 54

## What changed

- Add trial token validation and native inference receipt support.
- Wire out-of-control debt assessment into fold, render, and check gate.
- Implement structural CI YAML linting and DAG validation to retire debt.
- Implement child process execution and supervision to retire debt.
- Implement rate limiting and pacing gate to retire debt.
- Declare tier 1 for blobcommon, breathgate, childproc, and ciyaml.
- Implement blob utilities and tests to retire debt.
- Add tree status command and MCP inspection tool to isolate session diffs Closes.
- Add bounded native dense GPU-layer placement for 8 GiB Qwen3.8 parity.
- Fold negative group scales into packed int4 weights for Marlin.
- Implement Woodbury-Yamamoto-Fernandes chunkwise parallel recurrence.
- Add streaming SSE keepalive filter with repetition tripwire.
- Add cross-engine prompt token packet for AMD trials (#9883).
- Implement architecture preflight command and MCP tool Closes.
- Bind post-pass GPU clocks and spliced memory validation (#10281).
- Add deterministic execution-surface placement policy (#10687).
- Implement exclusive NPU model residency arbitration across concurrent sessions (#10686).
- Add shared rate limit cooldown controller across concurrent sessions.
- Add prompt-cache stable goal continuation session and goal skill.
- Enforce test immunity gate against implementation lane test tampers.
- Implement 10x working tree metrics with Cluster Landability Quotient and Half-Life.
- Add out-of-control detector and scoreboard debt page CLI.
- Add maturity debt lanes scanner, interest curve, and CLI.
- Add -tune MTP performance matrix and markdown renderer.
- Implement deterministic multi-task quality evaluation for speculative decode.
- Implement MTP speculative depth and threshold tuning sweep.
- Support prompt cache coexistence with MTP recurrent rollback.
- Add collective abort, constant upload, KV offload, recurrent rollback, and Vulkan graphs.
- Wire workspace tools, multi-turn context, and headless mode into fak chat #11126.
- Implement 16-row Direct I/O Safetensors row gather benchmark with CUDA stream memop synchronization (#11037).

## Reliability and correctness

- Wire test immunity check and add functions.exec_command recognizer support.
- Wire debt-lanes subcommand into score and maturity routers.
- Fix background process console window popups on Windows.
- Convert CUDA request-time panics into typed backend errors.

## Engineering quality and evidence

- Remove duplicate loopUsage and worktreeWorkerList definitions.
- Remove duplicate walk functions after extraction.
- Compact DSA/MLA latent cache with 16-token rolling reversion journal.
- Make hybrid CUDA decode graph-replay safe.
- Keep dense Q5_K/Q6_K directly resident on CUDA.
- Update commit-clean skill with isolated validation and review witness.
- Add concept disambiguation scorecard data rows.
- Add security fleet governance example output.
- Add hardware platforms, storage, and benchmark setup guides.
- Land benchmark machine profiles and hardware witness packets.
- Add debt-orchestrator multi-wave campaign skill.
- Add cross-validate adversarial audit skill.
- Land standalone architecture and concept notes.
- Split loop usage and worktree worker listing into dedicated files.
- Extract intent walking and rollup evaluation.
- Document super workstream coordinator and skill adapters.
- Fix 120s timeout in working tree inventory by batching checkpoints and bounding ignored files.
- Rename internal/ctxplans to internal/ctxplanlint.
- Rename internal/market to internal/marketplace to resolve sibling collision.
- Migrate baseline config path to configs/category-baselines.json.

## Upgrade and breaking changes

- No manual migration required unless specified above.
