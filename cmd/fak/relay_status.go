// relay_status.go — rung C7b of the perpetual-session relay track (#1900, epic #1860):
// `fak relay status <relay-id>`, the OFFLINE fold over a relay's shadow-baton legs. Where
// `fak relay resume` inspects ONE baton file, `status` folds the whole leg sequence a relay
// has emitted (the shadow-baton sidecars written by relay.EmitShadowBaton,
// shadow-baton-<relay-id>-leg<N>.json) into a per-leg view: for each leg its number, the
// tombstone reason it exited on, the SHA it observed, and its display-only note.
//
// Deliberately pure read/print in the relay_resume.go style: no reload re-verification, no
// resolver calls, no clock, no network. Each sidecar is decoded through the same C2 codec
// (relay.Parse) and schema-gated before any field is shown, so a non-baton in baton clothing
// cannot print as a leg. Cost is not a baton field today, so it is reported as `unknown`
// rather than invented — the done-condition is legs + reasons, and the no-`claimed`
// invariant forbids reporting a number the batons never carried.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/relay"
)

// relayLegStatus is one row of the folded status view: the display-only projection of a
// single shadow-baton sidecar. It carries pointers and labels only (leg number, the closed
// tombstone reason token, the observed SHA, the display note) — never trusted progress,
// mirroring the baton's own no-`claimed` invariant. It is the --json wire shape.
type relayLegStatus struct {
	Leg        int    `json:"leg"`
	Reason     string `json:"reason"`
	AtSHA      string `json:"at_sha"`
	Note       string `json:"note,omitempty"`
	NextAction string `json:"next_action,omitempty"`
	Source     string `json:"source"`
}

// runRelayStatus loads every shadow-baton sidecar for one relay id under --dir, schema-gates
// and folds them into a per-leg view sorted by leg number, and prints it — the aligned human
// summary by default, a canonical JSON array with --json. It returns the process exit code
// (0 ok, 1 a runtime error, 2 a usage error).
func runRelayStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("relay status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", ".", "directory holding the shadow-baton sidecars (shadow-baton-<relay-id>-leg<N>.json)")
	asJSON := fs.Bool("json", false, "emit the folded per-leg view as a canonical JSON array instead of the human summary")
	// A bare positional relay id is accepted in either order (`fak relay status RID --json`
	// or `... --json RID`), matching the repo's positional-leading verb convention.
	rest, positional := relayLeadingPositional(argv, false)
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	ids := fs.Args()
	if positional != "" {
		ids = append([]string{positional}, ids...)
	}
	if len(ids) != 1 {
		fmt.Fprintln(stderr, "fak relay status: exactly one relay id is required (fak relay status <relay-id> [--dir <path>] [--json])")
		return 2
	}
	relayID := ids[0]

	// The sidecar names are shadow-baton-<relay_id>-leg<N>.json (relay.EmitShadowBaton). A
	// glob over the relay id peels off exactly this relay's legs; every other file in the dir
	// is ignored.
	glob := filepath.Join(pathutil.ExpandTilde(*dir), fmt.Sprintf("shadow-baton-%s-leg*.json", relayID))
	matches, err := filepath.Glob(glob)
	if err != nil {
		fmt.Fprintf(stderr, "fak relay status: glob %s: %v\n", glob, err)
		return 1
	}

	legs := make([]relayLegStatus, 0, len(matches))
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak relay status: read %s: %v\n", path, err)
			return 1
		}
		b, err := relay.Parse(data)
		if err != nil {
			fmt.Fprintf(stderr, "fak relay status: %s: %v\n", path, err)
			return 1
		}
		// Reader contract step 3: reject any object that does not carry the exact schema tag
		// BEFORE trusting any other field, exactly as `fak relay resume` does.
		if b.Schema != relay.Schema {
			fmt.Fprintf(stderr, "fak relay status: %s: not a %s baton (schema %q)\n", path, relay.Schema, b.Schema)
			return 1
		}
		legs = append(legs, relayLegStatus{
			Leg:        b.Leg,
			Reason:     b.Tombstone.Reason,
			AtSHA:      b.Tombstone.AtSHA,
			Note:       b.Tombstone.Note,
			NextAction: b.NextAction,
			Source:     filepath.Base(path),
		})
	}
	// Legs are ordered by leg number — the relay's lineage order — not by filesystem glob
	// order, so the fold reads as the sequence the successor legs actually formed.
	sort.SliceStable(legs, func(i, j int) bool { return legs[i].Leg < legs[j].Leg })

	if *asJSON {
		if rc := encodeJSONOrFailPrefixed(stdout, stderr, legs, "fak relay status"); rc != 0 {
			return rc
		}
		return 0
	}
	printRelayStatus(stdout, relayID, legs)
	return 0
}

// printRelayStatus renders the folded per-leg view. An empty fold (no sidecars for the id)
// prints a clear "(no legs)" line rather than nothing, so an operator can tell "relay not
// found / not yet emitted" apart from a silent error. Cost is reported as `unknown` because
// no baton field records it.
func printRelayStatus(w io.Writer, relayID string, legs []relayLegStatus) {
	fmt.Fprintf(w, "relay %s  legs=%d  cost=unknown\n", relayID, len(legs))
	if len(legs) == 0 {
		fmt.Fprintf(w, "  (no legs) — no shadow-baton sidecars found for this relay id\n")
		return
	}
	for _, l := range legs {
		fmt.Fprintf(w, "  leg %d: %s @ %s%s\n", l.Leg, orNone(l.Reason), orNone(l.AtSHA), noteSuffix(l.Note))
		if strings.TrimSpace(l.NextAction) != "" {
			fmt.Fprintf(w, "    next_action: %s\n", l.NextAction)
		}
	}
}
