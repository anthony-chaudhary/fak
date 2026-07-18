package vcachegov

// The affinity-router tests were removed together with the router itself (#5190);
// see affinity.go for the cut rationale. The live prompt_cache_key derivation and
// its 32-char cap are pinned by TestResponsesPromptCacheKeyPresentAndStable in
// internal/agent.
