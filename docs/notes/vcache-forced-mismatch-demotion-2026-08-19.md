---
title: "Forced vCache mismatch demotion — 2026-08-19"
description: "Verdict: a live provider-cache false-warm result now produces a hash-chained demotion record that identifies the first turn where observed provider reuse..."
---
# Forced vCache mismatch demotion — 2026-08-19

**Verdict:** a live provider-cache false-warm result now produces a hash-chained demotion record that identifies the first turn where observed provider reuse diverged from the warm belief.

- `TestVCacheWarmthMetricsAndDemotionJournal` forces a warm → miss → warm sequence through the live gateway metrics path.
- The miss at turn 2 is classified `false_warm`; the reconstructed belief moves from `resident` to `expired`, so later decisions cannot continue treating the prefix as warm without new provider evidence.
- The journal records `action=mark_belief_cold`, `reason=false_warm`, `divergence_probe=provider_cache_read_boundary`, and `divergent_prefix_turn=2` in its hash chain.
- The same live path emits cumulative demotion count and false-warm-rate metrics. Correctness remains the full uncached prompt; the belief only governs cache economics.

Structured artifact: [`vcache-forced-mismatch-demotion-2026-08-19.json`](../_witnesses/vcache-forced-mismatch-demotion-2026-08-19.json).

This retires forced-mismatch demotion and first-divergence localization in #8090. The provider cannot expose byte-level internal cache divergence, so localization honestly names the first request-prefix boundary where cache-read evidence disagrees.
