package metrics

// device_spine_federate.go — the fleet-federation reader for the
// device-telemetry spine (issue #3237, epic #3236). ZML's api/client.zig
// scrapes peers' /metrics, parses with ignore_unknown_fields, marks
// remote=true, and appends into the same struct list — one pane-of-glass for
// free. This is the Go form of that mechanism, kept transport-agnostic: the
// caller fetches a peer's RenderJSON bytes however it likes (HTTP, file,
// pipe); ParsePeerSnapshot + Federate turn those bytes into rows of the SAME
// DeviceMetrics list every renderer (Prom/CSV/JSON/text) already reads, so
// federation costs no new surface.
//
// Generation: gen/next. Promotion evidence: a live scraper loop that polls
// peer /metrics endpoints and feeds Federate into the gateway's snapshot
// buffer. Demotion evidence: if the fleet never runs more than one host,
// federation is dead weight — retire the reader. Invalidating assumption:
// peers are assumed to speak the RenderJSON schema; a peer emitting only
// Prometheus text would need a Prom parser instead.

import (
	"encoding/json"
	"fmt"
)

// ParsePeerSnapshot parses a peer's RenderJSON output into federated rows:
// every row is marked Remote=true, and rows that do not already carry a Peer
// origin get this peer's name (a row that arrives with its own Peer keeps it,
// so multi-hop federation preserves the true origin). Unknown JSON fields are
// ignored — Go's json.Unmarshal skips them by default — so a newer peer with
// extra metrics still parses (ZML's ignore_unknown_fields). A JSON null or
// empty input yields an empty snapshot, not an error, mirroring RenderJSON's
// nil-renders-as-empty-array contract.
func ParsePeerSnapshot(data []byte, peer string) ([]DeviceMetrics, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var rows []DeviceMetrics
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, fmt.Errorf("peer %s snapshot: %w", peer, err)
	}
	for i := range rows {
		rows[i].Remote = true
		if rows[i].Peer == "" {
			rows[i].Peer = peer
		}
	}
	return rows, nil
}

// Federate appends peer snapshots onto a local snapshot into one
// pane-of-glass list. It is pure and deterministic — local rows first,
// unchanged, then each peer snapshot in argument order — and always returns a
// fresh slice, never mutating or aliasing its inputs, so a published Buffer
// snapshot can be federated without violating the read-only contract.
func Federate(local []DeviceMetrics, peers ...[]DeviceMetrics) []DeviceMetrics {
	n := len(local)
	for _, p := range peers {
		n += len(p)
	}
	out := make([]DeviceMetrics, 0, n)
	out = append(out, local...)
	for _, p := range peers {
		out = append(out, p...)
	}
	return out
}
