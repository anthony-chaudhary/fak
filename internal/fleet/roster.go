// Package fleet is the public, transport-agnostic core for operating a fleet of
// boxes — GPU servers, worker nodes — an operator drives over the private Slack
// control-bridge. It is the importable home for the typed roster, the per-box report
// seam schema, the deterministic fold + 0-100 readiness score, and the bounded
// render. Two binaries share it: cmd/fleetctl (the standalone control surface) and
// the `fak lab` verb (the fast local front door).
//
// THE PUBLIC/PRIVATE BOUNDARY IS A DATA CONTRACT, NOT A CODE IMPORT. The live control
// plane — the Slack control-bridge to the lab boxes — is private (it speaks a lab
// protocol and carries lab identifiers, so it lives in fak-private; see
// docs/gpu-server-private-boundary.md and docs/private-comms-channel.md). The seam between it
// and this public core is the per-box REPORT JSON (fak.fleet.report/v1, report.go):
// the private bridge emits one report file per box from live state; this package
// reads, folds, renders, and scores them. Neither side imports the other, and
// everything here is generic — a box id, a class, a state word, a version, an age —
// never a host, a channel, or a token.
package fleet

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// RosterSchema is the on-disk schema tag for a fleet roster file. A roster that
// names a DIFFERENT major is refused (fail-loud) so a future incompatible change
// can never be silently misread as the current one.
const RosterSchema = "fak.fleet.roster/v1"

// MaxBoxes caps a single roster. The whole tool is built and tested to stay
// readable and fast well past the 100-box target (render summarizes; the fold is
// linear); the cap is a sanity fence against a malformed file claiming millions of
// boxes, not a design ceiling.
const MaxBoxes = 4096

// Box is one controllable machine in the fleet — a GPU server, a worker node. It is
// deliberately GENERIC and transport-agnostic: nothing here names a lab host, a
// channel, or a token. Endpoint is an OPAQUE reference a transport resolves on its
// own terms (the public file transport treats it as a report-file stem; the private
// Slack/box bridge resolves it to a channel/session). Keeping the reference opaque
// is what lets the live control plane stay private while this core stays public —
// see docs/fleet.md and docs/gpu-server-private-boundary.md.
type Box struct {
	ID       string            `json:"id"`
	Class    string            `json:"class,omitempty"`    // hardware/role class, e.g. "a100x8", "h100x8", "cpu"
	Group    string            `json:"group,omitempty"`    // logical grouping, e.g. "lab-1", "us-west"
	Endpoint string            `json:"endpoint,omitempty"` // opaque transport ref; defaults to ID. KEEP GENERIC in a committed roster — never a real channel/session/host/token; the private bridge owns the id->channel map on its side.
	Labels   map[string]string `json:"labels,omitempty"`   // reserved authoring metadata; round-trips through `ls --json`, not yet consumed by a selector.
}

// Ref is the transport reference for a box: its Endpoint, or its ID when Endpoint is
// unset.
func (b Box) Ref() string {
	if b.Endpoint != "" {
		return b.Endpoint
	}
	return b.ID
}

// Roster is an ordered set of boxes. Authoring order is preserved so a render is
// stable; identity is by ID and must be unique.
type Roster struct {
	Schema string `json:"schema,omitempty"`
	Boxes  []Box  `json:"boxes"`
}

// LoadRoster parses a roster from JSON. It does NOT validate (call Validate for
// that), so a caller can load and then inspect a partially-bad file. Unknown fields
// are rejected — a typo'd key is a fail-loud error, never a silently-ignored field.
//
// Strict parsing means a roster is NOT forward-compatible: a future fleetctl that
// adds an optional roster field produces files an older binary rejects. That is the
// chosen trade — for an operator-authored local config a fail-loud "your fleetctl is
// stale" beats silently dropping a field it cannot honor.
func LoadRoster(r io.Reader) (Roster, error) {
	var ro Roster
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&ro); err != nil {
		return Roster{}, fmt.Errorf("decode roster: %w", err)
	}
	return ro, nil
}

// LoadRosterFile reads and parses a roster file.
func LoadRosterFile(path string) (Roster, error) {
	f, err := os.Open(path)
	if err != nil {
		return Roster{}, err
	}
	defer f.Close()
	return LoadRoster(f)
}

// Validate fails closed on a roster an operator cannot safely act on: a wrong schema
// major, no boxes, too many boxes, an empty/duplicate ID, an ID carrying whitespace or
// a path separator, or two boxes that resolve to the SAME report key (Ref()) — a silent
// collision that would fold one box's state in for two and skew the score. It returns
// ALL problems at once so a fix is one pass, not a guess-and-retry loop. An empty slice
// means the roster is valid.
//
// The ID guard keeps the ID a clean identity/display key and a safe DEFAULT report-file
// key. Endpoint stays OPAQUE here (a transport may resolve it however it likes); the
// file transport validates it for file-safety at read time (safeReportKey), so an
// escaping endpoint reads as an error, never an out-of-dir file.
func (ro Roster) Validate() []string {
	var probs []string
	if ro.Schema != "" && ro.Schema != RosterSchema {
		probs = append(probs, fmt.Sprintf("schema %q is not %q", ro.Schema, RosterSchema))
	}
	if len(ro.Boxes) == 0 {
		probs = append(probs, "roster has no boxes")
	}
	if len(ro.Boxes) > MaxBoxes {
		probs = append(probs, fmt.Sprintf("roster has %d boxes, the cap is %d", len(ro.Boxes), MaxBoxes))
	}
	seen := map[string]int{}
	seenRef := map[string]int{}
	for i, b := range ro.Boxes {
		switch {
		case b.ID == "":
			probs = append(probs, fmt.Sprintf("box[%d] has an empty id", i))
		case strings.TrimSpace(b.ID) != b.ID || strings.ContainsAny(b.ID, " \t/\\"):
			probs = append(probs, fmt.Sprintf("box[%d] id %q has whitespace or a path separator", i, b.ID))
		}
		if b.ID == "" {
			continue
		}
		if j, dup := seen[b.ID]; dup {
			probs = append(probs, fmt.Sprintf("box[%d] id %q duplicates box[%d]", i, b.ID, j))
			continue // a duplicate ID already collides on Ref(); don't double-report
		}
		seen[b.ID] = i
		key := b.Ref()
		if j, dup := seenRef[key]; dup {
			probs = append(probs, fmt.Sprintf("box[%d] resolves to the same report key %q as box[%d]", i, key, j))
		} else {
			seenRef[key] = i
		}
	}
	return probs
}

// Template builds a synthetic roster of n boxes in a single call — the "how do I add
// up to 100 boxes?" answer. IDs are zero-padded to a uniform width so they sort
// lexically in box order (box-001 .. box-100). It is the scaffold an operator edits,
// not a live roster.
func Template(n int, class, group, prefix string) Roster {
	if prefix == "" {
		prefix = "box"
	}
	width := len(fmt.Sprintf("%d", n))
	if width < 3 {
		width = 3
	}
	boxes := make([]Box, n)
	for i := 0; i < n; i++ {
		boxes[i] = Box{
			ID:    fmt.Sprintf("%s-%0*d", prefix, width, i+1),
			Class: class,
			Group: group,
		}
	}
	return Roster{Schema: RosterSchema, Boxes: boxes}
}
