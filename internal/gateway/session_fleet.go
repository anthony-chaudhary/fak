package gateway

// session_fleet.go — the live cross-MACHINE fleet aggregate, exposed on /debug/vars
// so an operator pane (fak guard's status area) can show the fleet-of-machines health
// beside the session-local blocks: how many operator boxes have published snapshots,
// how many are stale/needing-action, and a few rolled-up totals. It is the twin of
// SessionEndpoints (which shows THIS session's accounts+nodes) one level up — the whole
// fleet rather than the local seat.
//
// Like SetSessionEndpointsProvider, this is a PULL provider (called on each /debug/vars
// scrape) that the host — fak guard, which alone knows the machine roster on disk and
// the peers it has heard over the discovery spine — supplies. The default `fak serve`
// path never sets it, so the block is absent there. Unlike endpoints (a cheap FS glob),
// a fleet fold walks every published machine snapshot, so the host wraps its provider in
// a short TTL cache (see cmd/fak/guard_fleet.go); the gateway holds no cache and no
// policy, only the seam. Everything folded here is display metadata — machine ids,
// states, counts — never a token or a request payload, so the payload-free /debug/vars
// contract holds.

// SessionFleet is the cross-machine fleet aggregate: a verdict word, the machine count
// and the needs-attention split, rolled-up session/auth/version totals, and a bounded
// sample of per-machine rows. All fields are display-only; a cold operator with no peers
// folds to zero machines and the provider reports ok=false, so the block is omitted
// rather than emitted as "machines=0".
type SessionFleet struct {
	// Verdict is the fold's one-word health ("OK" | "ACTION" | "STALE" | "EMPTY" | ""),
	// rendered beside the fleet glyph. Empty is treated as unknown by the pane.
	Verdict string `json:"verdict,omitempty"`
	// Machines is how many boxes published a snapshot the fold saw.
	Machines int `json:"machines"`
	// Stale / Action are the needs-attention split across those machines: Stale is a box
	// whose snapshot aged past its republish horizon; Action is one that flagged a
	// decision. Their sum is the pane's "N need attention" count.
	Stale  int `json:"stale,omitempty"`
	Action int `json:"action,omitempty"`
	// Sessions is the total live sessions summed across the fleet; AuthBlocked and
	// VersionMismatches are the rolled-up trouble counts (seats that cannot serve, boxes
	// running a skewed fak version). Each is omitted when zero so a clean fleet is quiet.
	Sessions          int     `json:"sessions,omitempty"`
	AuthBlocked       int     `json:"auth_blocked,omitempty"`
	ThrottledSeats    int     `json:"throttled_seats,omitempty"`
	HealthySeats      int     `json:"healthy_seats,omitempty"`
	SeatCapacity      int     `json:"seat_capacity,omitempty"`
	ResumeBacklog     int     `json:"resume_backlog,omitempty"`
	HostLoad          float64 `json:"host_load,omitempty"`
	VersionMismatches int     `json:"version_mismatches,omitempty"`
	// Rows is a bounded, most-relevant sample of per-machine records (the host caps it so
	// a large fleet still folds into the aggregate totals rather than a wall of rows).
	Rows []SessionFleetMachine `json:"rows,omitempty"`
}

// SessionFleetMachine is one box in the fleet sample: its id, its fold state, the age of
// its published snapshot in minutes, its live session count, and the fak version it
// reported. Fields are zero-valued (and omitted) when the snapshot did not carry them.
type SessionFleetMachine struct {
	ID       string  `json:"id"`
	State    string  `json:"state,omitempty"`
	AgeMin   float64 `json:"age_min,omitempty"`
	Sessions int     `json:"sessions,omitempty"`
	Version  string  `json:"version,omitempty"`
}

// SetSessionFleetProvider installs the pull source for the live cross-machine fleet
// /debug/vars block. fak guard wires it to its TTL-cached snapshot fold (unioned with the
// discovery spine's live peers); the default serve path leaves it unset so the block is
// absent. Unlike the endpoints provider, the fleet provider reports its own ok — the host
// fold already knows when there is nothing to show (zero machines) — so the gateway does
// not second-guess it. Passing nil detaches it. Safe for concurrent use and on a nil
// Server.
func (s *Server) SetSessionFleetProvider(fn func() (SessionFleet, bool)) {
	if s == nil {
		return
	}
	s.sessionFleetMu.Lock()
	s.sessionFleetProvider = fn
	s.sessionFleetMu.Unlock()
}

// sessionFleet pulls the current fleet snapshot for /debug/vars. It returns ok=false when
// no provider is set OR the provider reports nothing to show (cold gateway / plain serve /
// an operator with no peers), so the debug block is omitted rather than emitted empty.
// SessionFleetProviderInstalled reports whether the host attached a fleet pull source.
// It exposes wiring state, not fleet data, and is safe for startup tests and diagnostics.
func (s *Server) SessionFleetProviderInstalled() bool {
	if s == nil {
		return false
	}
	s.sessionFleetMu.Lock()
	defer s.sessionFleetMu.Unlock()
	return s.sessionFleetProvider != nil
}

// SessionFleetSnapshot reads the currently installed fleet provider through the same
// omission rules used by /debug/vars. It is primarily a wiring witness for hosts: ok=false
// means no provider, no sample, or an empty fleet.
func (s *Server) SessionFleetSnapshot() (SessionFleet, bool) {
	return s.sessionFleet()
}

func (s *Server) sessionFleet() (SessionFleet, bool) {
	if s == nil {
		return SessionFleet{}, false
	}
	s.sessionFleetMu.Lock()
	fn := s.sessionFleetProvider
	s.sessionFleetMu.Unlock()
	if fn == nil {
		return SessionFleet{}, false
	}
	f, ok := fn()
	if !ok || f.Machines == 0 {
		return SessionFleet{}, false
	}
	return f, true
}
