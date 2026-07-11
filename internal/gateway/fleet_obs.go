package gateway

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

// fleetObsPathEnv names the file the gateway appends cross-trace fleet observations to.
// The launcher sets it to <repoRoot>/docs/nightrun/fleet-observations.jsonl on a guarded
// session so the always-on `fak knownbad correlate` fold and the emitting proxies agree on
// one path; leaving it unset disables the feed (the durable LIVELOCK journal row still
// lands — the feed is the CROSS-trace correlation input, not the per-trace witness).
const fleetObsPathEnv = "FAK_FLEET_OBS_PATH"

// fleetObsPath resolves the sink: the in-process test override wins, else the environment.
// Empty means the feed is disabled.
func (s *Server) fleetObsPath() string {
	if s != nil && strings.TrimSpace(s.fleetObsPathOverride) != "" {
		return strings.TrimSpace(s.fleetObsPathOverride)
	}
	return strings.TrimSpace(os.Getenv(fleetObsPathEnv))
}

// emitFleetObservation appends one guardrsi.FleetObservation as a JSONL line to the
// configured fleet-observation sink. It is best-effort and fail-open: a disabled feed, a
// marshal error, or an unwritable path is silently skipped, because the observation is a
// correlation HINT layered on top of the durable LIVELOCK journal row — never the record
// of record. Appends are serialized by fleetObsMu so concurrent traces cannot interleave a
// partial line, and the parent directory is created lazily so a first observation on a
// fresh checkout still lands.
func (s *Server) emitFleetObservation(obs guardrsi.FleetObservation) {
	if s == nil {
		return
	}
	path := s.fleetObsPath()
	if path == "" {
		return
	}
	line, err := json.Marshal(obs)
	if err != nil {
		return
	}

	s.fleetObsMu.Lock()
	defer s.fleetObsMu.Unlock()
	if dir := filepath.Dir(path); dir != "" {
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}
