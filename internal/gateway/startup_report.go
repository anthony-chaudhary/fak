package gateway

// This file is the on-demand home for the host's startup report. `fak guard` renders a
// full multi-line explanation of what now sits in front of the agent (gateway, upstream,
// floor, hooks, auth posture) — invaluable in a captured headless log, but a wall of
// text on an attended launch where the wrapped agent's full-screen TUI paints over it
// seconds later. So the host records the FULL text here regardless of how compact the
// terminal banner was, and the gateway serves it back for the life of the session as the
// startup_report field of /debug/vars — `fak info --startup` is the reader. The report
// is boot-time configuration prose, never request payload, so the /debug/vars
// payload-free contract holds.

// SetStartupReport records the full human-readable startup report the host rendered at
// boot. Passing "" clears it. Safe on a nil Server and for concurrent use; the text is
// stored on the same one-time boot profile the startup-phase gauges live on.
func (s *Server) SetStartupReport(text string) {
	if s == nil || s.startup == nil {
		return
	}
	s.startup.mu.Lock()
	s.startup.report = text
	s.startup.mu.Unlock()
}

// startupReportText returns the recorded startup report, or "" when the host never set
// one (a fak serve gateway, or a build predating the report wiring) — /debug/vars omits
// the field then, so "not recorded" stays distinguishable from an empty report.
func (s *Server) startupReportText() string {
	if s == nil || s.startup == nil {
		return ""
	}
	s.startup.mu.Lock()
	defer s.startup.mu.Unlock()
	return s.startup.report
}
