// Rung B3 (issue #1867): the shadow would-be-baton emitter. On a shadow signal
// (session's RelayShadowEvent, rung B2), project the shadow baton a rotation
// WOULD have written — pointers only, to a sidecar; do not act on it. The
// projector is pure (no I/O, no clock) in the fidelity.go style: a small input
// struct in, a Baton out; the emitter writes exactly one sidecar JSON file via
// the C2 codec and is read-only with respect to every durable store — no ledger
// row, no lease, no git object, no rotation is ever touched or triggered here
// (Track B — Observe).
package relay

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ShadowBatonInput is the durable inputs a projector reads from live leg
// state — every field is a durable pointer or a one-line label, never bytes
// and never a recap, mirroring the baton's own pointer-only invariant.
type ShadowBatonInput struct {
	// RelayID is the stable id for the whole relay.
	RelayID string
	// Leg is the current leg number (would-be closing leg).
	Leg int
	// ParentTrace is the trace id of the current leg.
	ParentTrace string
	// Objective is the active objective pin, carried VERBATIM into the baton.
	Objective ctxplan.ObjectivePin
	// DoneWhen is the one-line durable-store done predicate.
	DoneWhen string
	// NextAction is one line naming the next atomic action.
	NextAction string
	// StartSHA is the leg's git progress anchor.
	StartSHA string
	// LedgerRef optionally names the ledger row to re-read for verified progress.
	LedgerRef string
	// HeldRegion is the lease region (globs) a successor must re-acquire.
	HeldRegion []string
	// Artifacts are the durable pointers accumulated so far (refs, never bytes).
	Artifacts []Artifact
	// DoNotRederive are durable pointers to closed dead ends.
	DoNotRederive []string
	// OpenQuestions are durable pointers or short labels for unresolved decisions.
	OpenQuestions []string
	// ExitReason is the relay reason token, e.g. "RELAY_ARMED"; defaulted if empty.
	ExitReason string
	// AtSHA is the observed SHA; defaults to StartSHA if empty.
	AtSHA string
}

// ProjectShadowBaton assembles the baton a rotation WOULD write from the
// durable inputs in. It is PURE — no I/O, no clock, no store reads or writes —
// and sets no progress percentage and no recap: the Baton type has no such
// field, and this projector keeps it that way. The objective pin is carried
// verbatim, never rebuilt or re-digested.
func ProjectShadowBaton(in ShadowBatonInput) Baton {
	return Baton{
		Schema:      Schema,
		RelayID:     in.RelayID,
		Leg:         in.Leg,
		ParentTrace: in.ParentTrace,
		Objective:   in.Objective,
		DoneWhen:    in.DoneWhen,
		ProgressCursor: ProgressCursor{
			StartSHA:   in.StartSHA,
			LedgerRef:  in.LedgerRef,
			HeldRegion: nonNilStrings(in.HeldRegion),
		},
		NextAction:    in.NextAction,
		OpenQuestions: nonNilStrings(in.OpenQuestions),
		Artifacts:     append([]Artifact{}, in.Artifacts...),
		DoNotRederive: nonNilStrings(in.DoNotRederive),
		Tombstone: Tombstone{
			Reason: reasonOr(in.ExitReason, "RELAY_ARMED"),
			AtSHA:  firstNonEmpty(in.AtSHA, in.StartSHA),
		},
	}
}

// EmitShadowBaton projects the shadow baton from in, marshals it with the
// C2 codec, and writes it to ONE sidecar file under dir:
// shadow-baton-<relay_id>-leg<N>.json. It returns the written path. The
// sidecar is the only thing touched — no durable store is read or written and
// nothing acts on the projection.
func EmitShadowBaton(dir string, in ShadowBatonInput) (string, error) {
	b := ProjectShadowBaton(in)
	data, err := Marshal(b)
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, fmt.Sprintf("shadow-baton-%s-leg%d.json", in.RelayID, in.Leg))
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", fmt.Errorf("relay: emit shadow baton: %w", err)
	}
	return path, nil
}

// reasonOr returns s, or def when s is empty — the tombstone reason default.
func reasonOr(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// firstNonEmpty returns a when non-empty, else b — the observed-SHA default.
func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
