package vcachegov

// affinity.go previously held the cross-shard affinity router and its two safety
// guards (AffinityKey, AffinityHeader, HitSample, RehashDetector, BurstCap — issue
// #720, acceptance 3). The router was CUT in #5190 under the #2128 inert-code
// posture: it had zero non-test callers, so none of its autoscale-rehash detection
// or warming burst capping ever executed on a live path, while the glossary and
// wire-layer comments advertised it as a shipped mechanism.
//
// The LIVE cross-shard routing hint is the agent wire layer's own derivation
// (responsesPromptCacheKey in internal/agent/adapters.go), which hashes the
// request's cacheable head (model, system messages, tools) and owns the
// load-bearing 32-hex-char truncation — the Codex/ChatGPT backend 400s a
// prompt_cache_key longer than 64 chars. Re-introducing a governor-side router is
// a wire decision to be made with a real caller and an integration test proving it
// executes, not by resurrecting the deleted code.
