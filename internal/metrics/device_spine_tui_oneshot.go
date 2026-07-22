package metrics

// device_spine_tui_oneshot.go — the interactive TUI's non-interactive fallback
// (issue #4366, parent #3237, epic #3236). The fanout names "sharing
// RenderText's one-shot for a `fak status` pretty-print": the TUI must NOT grow
// a second aligned-table renderer, it must reuse the one-shot the spine already
// has. FleetView.OneShot is that single seam — it delegates straight to
// RenderText over the SAME snapshot, so a `fak status` pretty-print and the
// TUI's static fallback can never drift, and there is exactly one aligned-table
// code path to maintain.
//
// Generation: gen/next. Promotion evidence: a `fak status` verb that calls
// FleetView.OneShot for a one-shot and RenderOverview under a --watch ticker for
// the live grid. Demotion evidence: if `fak status` never ships, OneShot is a
// one-line delegate worth inlining away. Invalidating assumption: the one-shot
// and the TUI fallback are assumed to want the identical aligned table; a
// status view that needs a different summary shape would fork this seam.

// OneShot renders the fleet as the shared one-shot aligned table — the exact
// bytes RenderText produces over the same snapshot. It is the TUI's static
// (non-interactive) surface: no sparkline history, no grid layout, just the
// spine's canonical pretty-print, so `fak status` and the TUI never diverge on
// how a snapshot reads.
func (v *FleetView) OneShot(snapshot []DeviceMetrics) []byte {
	return RenderText(snapshot)
}
