package toolproc

// CompactJournal returns a bounded subset of events safe to persist as the new
// journal. It keeps every un-exited spawn (a live process whose liveness a
// future hook firing must still resolve, and which can be arbitrarily old) plus
// the last tailKeep events as a recent margin, preserving original order.
// Events of fully terminal CallIDs (a spawn paired with an exit or kill),
// outside the tail window, are dropped: they cannot affect any future identity,
// liveness, or bg-bridge resolution. This bounds the journal so the per-firing
// full parse stays O(bounded), not O(all-history) — the root cause behind the
// guard-hook O(n^2) latency regression (#3032).
func CompactJournal(events []Event, tailKeep int) []Event {
	if tailKeep < 0 {
		tailKeep = 0
	}
	if tailKeep >= len(events) {
		return events
	}
	terminal := make(map[string]bool)
	for _, ev := range events {
		if ev.Kind == EvExit || ev.Kind == EvKill {
			terminal[ev.CallID] = true
		}
	}
	tailStart := len(events) - tailKeep
	out := make([]Event, 0, len(events))
	for i, ev := range events {
		if i >= tailStart || (ev.Kind == EvSpawn && !terminal[ev.CallID]) {
			out = append(out, ev)
		}
	}
	return out
}
