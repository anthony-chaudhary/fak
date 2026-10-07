package perfledger

// ProbeMaxPromptTokens is the largest whole prompt (uncached + cached) a turn can
// carry and still count as a liveness probe. Router and health probes send a
// one-line prompt with no system prompt (4 uncached + 21 cached tokens on the
// Halos); a real agent or chat turn always carries a system prompt far above it.
// On two Halos, probes were 66-68% of perf rows; their warm, sub-second TTFT made
// the partial regime p50 read 148 ms when served turns took ~7 s (fak-private#3151).
const ProbeMaxPromptTokens = 32

// IsProbe reports whether the row is a liveness/probe turn that Summarize keeps
// out of the served-traffic statistics: a tiny prompt the server has already
// cached, because a probe repeats the same one-liner. A tiny cold prompt is
// still counted as served (a user's first short turn is real traffic).
func (r Record) IsProbe() bool {
	total := r.PromptTokens + r.CachedTokens
	return r.CachedTokens > 0 && total <= ProbeMaxPromptTokens
}
