package agent

// RetainedReleaseReporter is an optional on-demand backend observation. It is
// independent of request completion and must never be polled on token paths.
// The bool says the backend supports this reporter, not that its evidence is valid.
// Supported reporters preserve unavailable/invalid/stale states in their text.
type RetainedReleaseReporter interface {
	RetainedReleaseSummary() (string, bool)
}

// RetainedReleaseSummary forwards to the immutable serving backend without
// holding planner/request locks. The backend owns serialization of its query.
func (p *InKernelPlanner) RetainedReleaseSummary() (string, bool) {
	if p == nil {
		return "", false
	}
	// Prefer one neutral observation, including empty or unavailable text.
	// The backend contract returns text; the planner adds the support bit.
	if reporter, ok := p.backend.(interface{ RetainedReleaseSummary() string }); ok {
		return reporter.RetainedReleaseSummary(), true
	}
	reporter, ok := p.backend.(interface{ VulkanRetainedReleaseSummary() string })
	if !ok {
		return "", false
	}
	return reporter.VulkanRetainedReleaseSummary(), true
}
