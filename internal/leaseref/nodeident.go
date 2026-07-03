package leaseref

// nodeident.go is the NODE IDENTITY convention (#2304, epic #2254) — the identity
// layer under the planes: holder = "<node-id>/<session-id>", where node-id is
// STABLE PER MACHINE (hostname-keyed, bound to the hardware catalog entry in
// experiments/benchmark/catalog.json) and session-id is the leaseref session
// descriptor id (session.go). Record.Holder was free-form; this file is the ONE
// helper that mints and parses the convention so every writer (acquire, session
// publish, intent claim, dev-server, GH announce) can share it and a reader can
// answer "WHICH MACHINE holds this?" without guessing. Session heartbeats (#2164)
// give liveness; the node component gives locality.
//
// TOLERANCE IS LOAD-BEARING: the fleet's history is full of free-form holders
// ("A:1", "host:pid", a bare name). ParseHolder NEVER errors — a holder that does
// not carry the convention classifies NodeUnknown, exactly like the liveness
// fail-closed rule treats an unbound session. The convention is adopted by
// writers going forward, never enforced retroactively on stored records: nothing
// here changes the Record JSON shape or the bytes a legacy blob round-trips.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// NodeUnknown is the node classification for a holder that does not carry the
// <node-id>/<session-id> convention (a legacy free-form holder, or an empty /
// unresolvable hostname). It is a CLASSIFICATION, never an error — the same
// absence-is-not-evidence posture as LivenessPeerUnknown.
const NodeUnknown = "node-unknown"

// CatalogRelPath is the repo-relative hardware catalog the node id binds to: its
// machines table maps a stable machine id to the hostname it answers as.
const CatalogRelPath = "experiments/benchmark/catalog.json"

// holderSep splits the two components of a conventional holder. Each component is
// one validID-safe segment, so a conventional holder contains EXACTLY one '/'.
const holderSep = "/"

// HolderIdentity is the parsed form of a Record.Holder. Node is the stable
// machine id (NodeUnknown when the holder is legacy free-form), Session is the
// leaseref session descriptor id ("" when unknown), Raw is the holder string as
// recorded — always preserved, so no information is lost by parsing.
type HolderIdentity struct {
	Node    string `json:"node"`
	Session string `json:"session,omitempty"`
	Raw     string `json:"raw"`
}

// Structured reports whether the holder carried the <node-id>/<session-id>
// convention — i.e. whether Node names a real machine rather than NodeUnknown.
func (h HolderIdentity) Structured() bool { return h.Node != NodeUnknown }

// MintHolder mints the conventional holder string "<node-id>/<session-id>". Each
// component is sanitized to one validID-safe segment (so a minted holder always
// parses back Structured), and an empty/unsanitizable component degrades to the
// NodeUnknown / "unknown" placeholder rather than failing — writers publish
// best-effort and a mint must never block them.
func MintHolder(nodeID, sessionID string) string {
	n := sanitizeHolderSegment(nodeID)
	if n == "" {
		n = NodeUnknown
	}
	s := sanitizeHolderSegment(sessionID)
	if s == "" {
		s = "unknown"
	}
	return n + holderSep + s
}

// ParseHolder classifies a Record.Holder against the convention. It NEVER errors:
// a holder that is not exactly <segment>/<segment> with both segments validID-safe
// is a legacy free-form holder and classifies {Node: NodeUnknown, Session: ""},
// with Raw preserved either way.
func ParseHolder(holder string) HolderIdentity {
	h := HolderIdentity{Node: NodeUnknown, Raw: holder}
	parts := strings.Split(holder, holderSep)
	if len(parts) != 2 || !validID(parts[0]) || !validID(parts[1]) {
		return h
	}
	h.Node, h.Session = parts[0], parts[1]
	return h
}

// HolderNode is the read-side convenience every surface uses: the node component
// of the record's holder — the stable machine id when the holder follows the
// convention, NodeUnknown for a legacy free-form holder.
func (r Record) HolderNode() string { return ParseHolder(r.Holder).Node }

// ResolveNodeID resolves a hostname to its stable node id: the hardware-catalog
// machine id when the (case-insensitively keyed) hostname is cataloged, else the
// sanitized hostname itself — still stable per machine, just not catalog-bound.
// A blank/unsanitizable hostname resolves NodeUnknown. Pure — hostToID is the
// caller's catalog view (see LoadCatalogNodeIDs), so tests drive it with literals.
func ResolveNodeID(hostname string, hostToID map[string]string) string {
	h := strings.ToLower(strings.TrimSpace(hostname))
	if h == "" {
		return NodeUnknown
	}
	if id, ok := hostToID[h]; ok && id != "" {
		return id
	}
	if s := sanitizeHolderSegment(h); s != "" {
		return s
	}
	return NodeUnknown
}

// LoadCatalogNodeIDs reads the hardware catalog's machines table and returns the
// lowercased-hostname -> machine-id map ResolveNodeID keys on. Only the two
// fields the binding needs are decoded; unknown fields are ignored, so the
// catalog schema can grow without touching this reader.
func LoadCatalogNodeIDs(path string) (map[string]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Machines map[string]struct {
			Hostname string `json:"hostname"`
		} `json:"machines"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	out := make(map[string]string, len(doc.Machines))
	for id, m := range doc.Machines {
		host := strings.ToLower(strings.TrimSpace(m.Hostname))
		if host == "" || id == "" {
			continue
		}
		out[host] = id
	}
	return out, nil
}

// LocalNodeID resolves THIS machine's stable node id: os.Hostname() keyed against
// the hardware catalog under repoRoot ("" = the process cwd). Best-effort by
// design — a missing/unreadable catalog or an uncataloged hostname falls back to
// the sanitized hostname (still stable), and no hostname at all is NodeUnknown.
func LocalNodeID(repoRoot string) string {
	host, _ := os.Hostname()
	if repoRoot == "" {
		repoRoot = "."
	}
	hostToID, err := LoadCatalogNodeIDs(filepath.Join(repoRoot, filepath.FromSlash(CatalogRelPath)))
	if err != nil {
		hostToID = nil
	}
	return ResolveNodeID(host, hostToID)
}

// sanitizeHolderSegment coerces s into one validID-safe segment: every byte
// outside [A-Za-z0-9._-] becomes '-', ref-illegal leading '-'/'.' runs are
// trimmed, and the segment is capped at validID's 200-byte bound. Returns "" when
// nothing safe remains — the caller picks the honest placeholder.
func sanitizeHolderSegment(s string) string {
	s = strings.TrimSpace(s)
	b := make([]byte, 0, len(s))
	for _, c := range []byte(s) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '-', c == '_', c == '.':
			b = append(b, c)
		default:
			b = append(b, '-')
		}
	}
	out := strings.TrimLeft(string(b), "-.")
	if len(out) > 200 {
		out = out[:200]
	}
	return out
}
